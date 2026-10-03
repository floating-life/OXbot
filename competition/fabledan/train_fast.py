# -*- coding: utf-8 -*-
"""Dual-GPU DMC trainer: central batched GPU inference (DanLM-style).

Processes:
  - N actor processes (CPU): run `ring` concurrent rounds each via RingRunner,
    send batched decision requests to the inference server, push finished
    episodes to the learner.
  - 1 inference server process (GPU `--infer-device`): batches all pending
    decisions across actors into single forward passes; refreshes weights
    every cycle.
  - main process = learner (GPU `--device`): replay buffer + gradient steps,
    checkpointing, paired eval vs rule baseline & frozen snapshot, and NumPy
    export. Run the pre-upload check separately to validate and package a bot.

Typical run on the supplied 9800X3D + RTX 5080 preset:
    scripts\train_5080.bat
Resume:
    python -m fabledan.train_fast --out ckpts/fast1 --resume ckpts/fast1/latest.pt
"""

import argparse
import os
import queue as pyqueue
import math
import signal
import time

import numpy as np

try:
    import torch
    import torch.multiprocessing as mp
except ImportError as e:
    raise SystemExit("PyTorch required: pip install torch") from e

from .model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt
from .encode import FEAT_DIM, STRUCTURE_FEAT_DIM
from .train import DMC_REWARD_SCHEMA, Replay, initialize_training


# ---------------------------------------------------------------------------
# actor process
# ---------------------------------------------------------------------------

def actor_proc(actor_id, req_q, resp_q, sample_q, stop_ev, ring, eps, top_k, seed,
               ladder_frac=0.0, feature_dim=FEAT_DIM):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    from .ring import RingRunner
    runner = RingRunner(ring=ring, seed=seed, eps=eps, top_k=top_k,
                        ladder_frac=ladder_frac, feature_dim=feature_dim)
    while not stop_ev.is_set():
        reqs = runner.collect_requests()
        # one message per actor: (actor_id, [(slot, toks, feats), ...])
        req_q.put((actor_id, reqs))
        # Keep one request outstanding per actor. Resending after a timeout
        # would apply a delayed response to a different game position.
        while not stop_ev.is_set():
            try:
                results = resp_q.get(timeout=1)
                break
            except pyqueue.Empty:
                continue
        else:
            return
        runner.step(dict(results))
        eps_out = runner.pop_episodes()
        if eps_out:
            sample_q.put(eps_out)


# ---------------------------------------------------------------------------
# inference server process
# ---------------------------------------------------------------------------

def infer_server(cfg_dict, req_q, resp_qs, weight_q, stop_ev, device,
                 max_decisions, stats_every):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = str(device).startswith("cuda")
    cfg = ModelConfig.from_dict(cfg_dict)
    model = FableDanNet(cfg).to(device).eval()
    sd = weight_q.get()
    model.load_state_dict(sd)
    n_req = 0
    n_dec = 0
    n_fwd = 0
    t_last = time.time()
    while not stop_ev.is_set():
        # weight refresh (non-blocking)
        try:
            while True:
                sd = weight_q.get_nowait()
                model.load_state_dict(sd)
        except pyqueue.Empty:
            pass
        # gather a batch of actor messages
        msgs = []
        ndec = 0
        try:
            m = req_q.get(timeout=0.5)
            msgs.append(m)
            ndec += len(m[1])
            while ndec < max_decisions:
                m = req_q.get_nowait()
                msgs.append(m)
                ndec += len(m[1])
        except pyqueue.Empty:
            pass
        if not msgs:
            continue
        # build batch
        all_toks, all_feats, counts, route = [], [], [], []
        for actor_id, reqs in msgs:
            for slot, toks, feats in reqs:
                all_toks.append(toks)
                all_feats.append(feats)
                counts.append(feats.shape[0])
                route.append((actor_id, slot))
        B = len(all_toks)
        maxlen = max(len(t) for t in all_toks)
        T = np.zeros((B, maxlen), dtype=np.int64)
        L = np.zeros(B, dtype=np.int64)
        for j, t in enumerate(all_toks):
            T[j, :len(t)] = t
            L[j] = len(t)
        Fcat = np.concatenate(all_feats, axis=0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=use_amp):
            tT = torch.from_numpy(T).to(device, non_blocking=True)
            tL = torch.from_numpy(L).to(device, non_blocking=True)
            tF = torch.from_numpy(Fcat).to(device, non_blocking=True)
            ctx, _ = model.encode_seq(tT, tL)
            cnt = torch.tensor(counts, device=device)
            ctx_rep = ctx.repeat_interleave(cnt, dim=0)
            hemb = model.hand_mlp(tF)
            q = model.q_head(torch.cat([ctx_rep, hemb], dim=-1))[:, 0]
            q = q.float().cpu().numpy()
        # split and route
        per_actor = {}
        off = 0
        for (actor_id, slot), c in zip(route, counts):
            per_actor.setdefault(actor_id, []).append((slot, q[off:off + c]))
            off += c
        for actor_id, results in per_actor.items():
            resp_qs[actor_id].put(results)
        n_req += len(msgs)
        n_dec += B
        n_fwd += 1
        if time.time() - t_last > stats_every:
            print("[infer] %.0f decisions/s, avg batch %.0f decisions "
                  "(%.1f actor msgs)" % (n_dec / (time.time() - t_last),
                                         n_dec / max(n_fwd, 1),
                                         n_req / max(n_fwd, 1)), flush=True)
            n_req = n_dec = n_fwd = 0
            t_last = time.time()


# ---------------------------------------------------------------------------
# packaging (auto botzone zip)
# ---------------------------------------------------------------------------

def pack_botzone_zip(weights_npz, out_zip):
    """Code zip (+ versioned weights copy next to it, see packaging.py).
    Returns a human-readable summary line."""
    from .packaging import pack
    zp, wcopy, wname = pack(weights_npz, out_zip)
    return "%s + %s" % (zp, wcopy)


# ---------------------------------------------------------------------------
# learner / main
# ---------------------------------------------------------------------------

def run_eval(model, device, games, opponent="rule", opp_model=None, seed=123,
             ladder_frac=0.0, check_stop=None):
    """Duplicate evaluation (same deal twice, seats swapped)."""
    from .agents import RuleAgent, TorchAgent
    from .evaluate import evaluate
    make_b = (lambda: RuleAgent()) if opponent == "rule" else \
        (lambda: TorchAgent(opp_model, device=device))
    wr, avg = evaluate(lambda: TorchAgent(model, device=device), make_b,
                       games=games, seed=seed, duplicate=True,
                       ladder_frac=ladder_frac, check_stop=check_stop)
    return wr, avg


def validate_args(ap, args):
    for name in ("actors", "ring", "cycles", "buffer", "diversity",
                 "steps_per_cycle", "batch", "micro_batch", "eval_cycles",
                 "eval_games", "ckpt_cycles", "max_decisions", "top_k"):
        if getattr(args, name) <= 0:
            ap.error("--%s must be positive" % name.replace("_", "-"))
    for name in ("snapshot_cycles", "export_cycles"):
        if getattr(args, name) < 0:
            ap.error("--%s must be nonnegative" % name.replace("_", "-"))
    for name in ("eps", "ladder_frac"):
        if not 0 <= getattr(args, name) <= 1:
            ap.error("--%s must be between 0 and 1" % name.replace("_", "-"))
    for name in ("belief_weight", "max_hours"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            ap.error("--%s must be finite and nonnegative" % name.replace("_", "-"))
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        ap.error("--lr must be finite and positive")
    if args.ntp_weight is not None and (not math.isfinite(args.ntp_weight)
                                        or args.ntp_weight < 0):
        ap.error("--ntp-weight must be finite and nonnegative")
    if args.n_blocks is not None and args.n_blocks <= 0:
        ap.error("--n-blocks must be positive")
    if args.buffer < args.diversity:
        ap.error("--buffer must be at least --diversity")
    if args.eval_games % 2:
        ap.error("--eval-games must be even for duplicate evaluation")
    if args.seed < 0:
        ap.error("--seed must be nonnegative")
    for name in ("device", "infer_device"):
        value = getattr(args, name)
        try:
            dev = torch.device(value)
        except (RuntimeError, ValueError) as exc:
            ap.error("--%s: %s" % (name.replace("_", "-"), exc))
        if dev.type not in ("cpu", "cuda"):
            ap.error("--%s must select cpu or cuda" % name.replace("_", "-"))
        if dev.type == "cuda" and (not torch.cuda.is_available()
                                  or (dev.index is not None and dev.index >= torch.cuda.device_count())):
            ap.error("--%s selects an unavailable CUDA device" % name.replace("_", "-"))


def atomic_checkpoint(model, opt, meta, path):
    """Keep the last valid checkpoint if saving is interrupted."""
    temporary = path + ".tmp"
    try:
        save_ckpt(model, opt, meta, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_export(model, path):
    temporary = path + ".tmp.npz"
    try:
        export_npz(model, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class TrainingDeadline(Exception):
    pass


def check_workers(workers):
    failed = [(p.name, p.exitcode) for p in workers if not p.is_alive()]
    if failed:
        raise RuntimeError("training worker stopped: %s" % failed)


def shutdown_workers(stop_ev, workers, queues):
    stop_ev.set()
    deadline = time.monotonic() + 5
    for p in workers:
        p.join(timeout=max(0, deadline - time.monotonic()))
    for p in workers:
        if p.is_alive():
            p.terminate()
    for p in workers:
        p.join(timeout=1)
    for q in queues:
        q.cancel_join_thread()
        q.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ckpts/fast1")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--resume", default="", help="continue a DMC run, including optimizer and replay")
    start.add_argument("--warm-start", default="", help="start a new DMC run from BC or other compatible weights")
    ap.add_argument("--warm-start-q-scale", type=float, default=None,
                    help="positive final-Q scale at warm start (BC: 0.05; other: 1); not Q calibration")
    ap.add_argument("--feature-dim", type=int, choices=(FEAT_DIM, STRUCTURE_FEAT_DIM), default=None,
                    help="feature dimension (default: source checkpoint or 80); migrate only via warm start")
    ap.add_argument("--actors", type=int, default=max(4, (os.cpu_count() or 8) - 6))
    ap.add_argument("--ring", type=int, default=16)
    ap.add_argument("--cycles", type=int, default=10**9)
    ap.add_argument("--buffer", type=int, default=262144)
    ap.add_argument("--diversity", type=int, default=2)
    ap.add_argument("--steps-per-cycle", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=256,
                    help="gradient-accumulation chunk size (memory bound)")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (new run: 1e-4; resume: saved value)")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--ntp-weight", type=float, default=None,
                    help="NTP loss weight (new run: 0.02; resume: saved config)")
    ap.add_argument("--belief-weight", type=float, default=0.05,
                    help="aux loss: predict opponents' hidden hands (0=off)")
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="transformer blocks (new run: 4; resume: saved config)")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--eval-cycles", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--ckpt-cycles", type=int, default=5)
    ap.add_argument("--snapshot-cycles", type=int, default=500,
                    help="frozen self snapshot refresh for shadow eval")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1
                    else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--infer-device", default="cuda:0"
                    if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-decisions", type=int, default=1024,
                    help="max decisions per inference batch")
    ap.add_argument("--ladder-frac", type=float, default=0.0,
                    help="fraction of self-play rounds in the Botzone default "
                         "setting (level 2, no tribute); rest random")
    ap.add_argument("--export-cycles", type=int, default=50,
                    help="every N cycles export latest.npz (0 = periodic export off)")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop after N hours and save checkpoint + npz "
                         "(0 = no time limit)")
    args = ap.parse_args()
    args.trainer = "train_fast"
    validate_args(ap, args)
    torch.set_num_threads(2)
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = str(device).startswith("cuda")
    torch.manual_seed(args.seed)
    try:
        model, opt, previous_meta, startup = initialize_training(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))
    cfg = model.cfg
    start_cycle = previous_meta.get("cycle", 0)
    total_samples = previous_meta.get("total_samples", 0)
    if args.resume:
        print("resumed cycle %d, %d samples" % (start_cycle, total_samples))
    elif args.warm_start:
        print("warm start from %s; optimizer/counters reset, Q output scale=%g" % (
            startup["source_training_kind"], startup["q_output_scale"]), flush=True)
    if args.cycles <= start_cycle:
        ap.error("--cycles is the total target; it must exceed resumed cycle %d" % start_cycle)
    replay = Replay(args.buffer, belief_dim=45 if args.belief_weight > 0 else 0,
                    feat_dim=cfg.feat_dim)
    if "replay" in previous_meta:
        try:
            replay.load_state_dict(previous_meta["replay"])
        except (ValueError, KeyError) as exc:
            ap.error(str(exc))
        # Drop the loaded arrays now that Replay owns validated copies.
        previous_meta = dict(previous_meta)
        previous_meta.pop("replay")
        print("restored %d replay samples" % replay.n, flush=True)
    elif args.resume:
        print("legacy checkpoint has no replay; collecting fresh samples before learning", flush=True)
    os.makedirs(args.out, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    req_q = mp.Queue(maxsize=4096)
    sample_q = mp.Queue(maxsize=4096)
    weight_q = mp.Queue(maxsize=4)
    resp_qs = [mp.Queue(maxsize=8) for _ in range(args.actors)]
    stop_ev = mp.Event()

    def push_weights():
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        try:
            weight_q.put_nowait(sd)
        except pyqueue.Full:
            pass
    rng = np.random.default_rng(args.seed)
    if "sample_rng" in previous_meta:
        rng.bit_generator.state = previous_meta["sample_rng"]
    if "torch_rng" in previous_meta:
        torch.set_rng_state(previous_meta["torch_rng"].cpu())
    if torch.cuda.is_available() and "cuda_rng" in previous_meta:
        torch.cuda.set_rng_state_all([state.cpu() for state in previous_meta["cuda_rng"]])
    cycle_budget = args.buffer // args.diversity
    best_wr = previous_meta.get("best_wr", 0.0)
    milestone_95 = previous_meta.get("milestone_95", False)
    old_settings = previous_meta.get("training_args", {})
    if "ladder_frac" in old_settings and old_settings["ladder_frac"] != args.ladder_frac:
        best_wr, milestone_95 = 0.0, False
        print("Evaluation distribution changed; resetting best-win-rate tracking.", flush=True)
    snapshot = None
    snapshot_cycle = previous_meta.get("snapshot_cycle", start_cycle)
    if args.snapshot_cycles:
        snapshot = FableDanNet(cfg).to(device).eval()
        snapshot.load_state_dict(previous_meta.get("snapshot_model", model.state_dict()))
    t_start = time.monotonic()
    initial_samples = total_samples
    completed_cycle = start_cycle
    optimizer_steps = previous_meta.get("optimizer_steps", 0)
    partial_steps = 0
    stop_reason = "cycles completed"
    workers = []
    queues = [req_q, sample_q, weight_q] + resp_qs

    def metadata():
        data = {"training_kind": "dmc", "trainer": "train_fast",
                "feature_dim": cfg.feat_dim, "feature_version": cfg.feature_version,
                "checkpoint_version": 2, "reward_schema": DMC_REWARD_SCHEMA,
                "objective": "MSE to completed-round team score / 3 plus auxiliary NTP/belief losses",
                "initialization": previous_meta.get("initialization", startup),
                "startup": startup, "replay": replay.state_dict(),
                "resume_scope": "model, optimizer, replay and learner RNG; actors restart new rounds",
                "cycle": completed_cycle, "total_samples": total_samples,
                "best_wr": best_wr, "milestone_95": milestone_95,
                "optimizer_steps": optimizer_steps,
                "partial_cycle_steps": partial_steps,
                "elapsed_seconds": previous_meta.get("elapsed_seconds", 0)
                                   + time.monotonic() - t_start,
                "sample_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "training_args": vars(args).copy(), "stop_reason": stop_reason}
        if torch.cuda.is_available():
            data["cuda_rng"] = torch.cuda.get_rng_state_all()
        if snapshot is not None:
            data["snapshot_model"] = {k: v.detach().cpu().clone()
                                      for k, v in snapshot.state_dict().items()}
            data["snapshot_cycle"] = snapshot_cycle
        return data

    def check_deadline():
        if args.max_hours and time.monotonic() - t_start >= args.max_hours * 3600:
            raise TrainingDeadline()

    try:
        server = mp.Process(target=infer_server, name="inference",
                            args=(cfg.to_dict(), req_q, resp_qs, weight_q,
                                  stop_ev, args.infer_device, args.max_decisions,
                                  30.0), daemon=True)
        server.start()
        workers.append(server)
        push_weights()
        for a in range(args.actors):
            p = mp.Process(target=actor_proc, name="actor-%d" % a,
                           args=(a, req_q, resp_qs[a], sample_q, stop_ev,
                                 args.ring, args.eps, args.top_k,
                                 args.seed + a + start_cycle * args.actors,
                                 args.ladder_frac, cfg.feat_dim), daemon=True)
            p.start()
            workers.append(p)

        for cycle in range(start_cycle, args.cycles):
            got = 0
            partial_steps = 0
            t0 = time.monotonic()
            while got < cycle_budget:
                check_deadline()
                check_workers(workers)
                try:
                    ep = sample_q.get(timeout=1)
                except pyqueue.Empty:
                    continue
                for toks, feat, z, bel in ep:
                    replay.add(toks, feat, z, bel)
                got += len(ep)
                total_samples += len(ep)
            t_collect = time.monotonic() - t0
            t0 = time.monotonic()
            model.train()
            losses = []
            mb = min(args.micro_batch, args.batch)
            n_chunks = (args.batch + mb - 1) // mb
            for _ in range(args.steps_per_cycle):
                check_deadline()
                check_workers(workers)
                T0, L0, F0, Z0, BEL0 = replay.sample(args.batch, rng)
                opt.zero_grad(set_to_none=True)
                step_loss = 0.0
                for c in range(n_chunks):
                    s, e = c * mb, min((c + 1) * mb, args.batch)
                    T = torch.from_numpy(T0[s:e]).to(device)
                    L = torch.from_numpy(L0[s:e]).to(device)
                    F = torch.from_numpy(F0[s:e]).to(device).unsqueeze(1)
                    Z = torch.from_numpy(Z0[s:e]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        q, hid = model(T, L, F)
                        loss_q = torch.nn.functional.mse_loss(q[:, 0].float(), Z)
                        loss = loss_q
                        if cfg.ntp_weight > 0:
                            loss = loss + cfg.ntp_weight * model.ntp_loss(T, hid)
                        if args.belief_weight > 0 and BEL0 is not None:
                            idx_last = (L - 1).clamp(min=0)
                            ctx = hid[torch.arange(hid.shape[0], device=device), idx_last]
                            bel_t = torch.from_numpy(BEL0[s:e]).to(device)
                            loss = loss + args.belief_weight * model.belief_loss(ctx.float(), bel_t)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError("non-finite DMC loss; refusing optimizer update")
                    ((e - s) / args.batch * loss).backward()
                    step_loss += loss_q.item() * (e - s) / args.batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                optimizer_steps += 1
                partial_steps += 1
                losses.append(step_loss)
            model.eval()
            completed_cycle = cycle + 1
            partial_steps = 0
            t_train = time.monotonic() - t0
            push_weights()
            sps = (total_samples - initial_samples) / (time.monotonic() - t_start + 1e-9)
            print("cycle %d  samples %s  loss %.4f  collect %.1fs train %.1fs  "
                  "%.0f samples/s" % (completed_cycle, f"{total_samples:,}",
                                      float(np.mean(losses)), t_collect, t_train, sps), flush=True)

            if completed_cycle % args.eval_cycles == 0:
                check_deadline()
                wr, avg = run_eval(model, device, args.eval_games, "rule",
                                   ladder_frac=args.ladder_frac, check_stop=check_deadline)
                msg = "  eval vs rule: %.1f%% (avg reward %+.2f)" % (wr * 100, avg)
                if snapshot is not None:
                    check_deadline()
                    wr_s, avg_s = run_eval(model, device, args.eval_games, "self",
                                           opp_model=snapshot, ladder_frac=args.ladder_frac,
                                           check_stop=check_deadline)
                    msg += "  vs snapshot(-%d cyc): %.1f%% (avg reward %+.2f)" % (
                        completed_cycle - snapshot_cycle, wr_s * 100, avg_s)
                print(msg, flush=True)
                if wr >= best_wr:
                    best_wr = wr
                    atomic_checkpoint(model, opt, metadata(), os.path.join(args.out, "best.pt"))
                    atomic_export(model, os.path.join(args.out, "best.npz"))
                if wr >= 0.95 and not milestone_95:
                    milestone_95 = True
                    print("  milestone: vs rule >= 95%; run scripts\\check_before_upload.bat "
                          "with the exported weights before uploading.", flush=True)
                    atomic_export(model, os.path.join(args.out, "latest.npz"))
            # Refresh AFTER evaluation so the model is compared with the old snapshot.
            if args.snapshot_cycles and completed_cycle % args.snapshot_cycles == 0:
                snapshot = FableDanNet(cfg).to(device).eval()
                snapshot.load_state_dict(model.state_dict())
                snapshot_cycle = completed_cycle
            if completed_cycle % args.ckpt_cycles == 0:
                atomic_checkpoint(model, opt, metadata(), os.path.join(args.out, "latest.pt"))
            if args.export_cycles and completed_cycle % args.export_cycles == 0:
                atomic_export(model, os.path.join(args.out, "latest.npz"))
            check_deadline()
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nInterrupted; saving resumable checkpoint and NumPy weights.", flush=True)
    except TrainingDeadline:
        stop_reason = "time limit reached"
        print("\nTime limit reached; saving checkpoint and NumPy weights.", flush=True)
    except Exception:
        stop_reason = "training failed"
        raise
    finally:
        try:
            shutdown_workers(stop_ev, workers, queues)
        finally:
            model.eval()
            atomic_checkpoint(model, opt, metadata(), os.path.join(args.out, "latest.pt"))
            atomic_export(model, os.path.join(args.out, "latest.npz"))
            print("Saved cycle %d (%s): %s and %s" % (
                completed_cycle, stop_reason, os.path.join(args.out, "latest.pt"),
                os.path.join(args.out, "latest.npz")), flush=True)


if __name__ == "__main__":
    main()
