# -*- coding: utf-8 -*-
"""DMC self-play training (PyTorch). Run on your GPU machine.

Cycle-based (DanLM-style): each cycle actors collect fresh samples with the
latest weights, learner does S gradient steps on the FIFO replay buffer,
then weights are broadcast to actors.

Usage (Windows/Linux):
    python -m fabledan.train --out ckpts/run1
    python -m fabledan.train --out ckpts/run1 --resume ckpts/run1/latest.pt
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import queue as pyqueue
import random
import time

import numpy as np

try:
    import torch
    import torch.multiprocessing as mp
except ImportError as e:
    raise SystemExit("PyTorch required for training: pip install torch") from e

from .encode import FEAT_DIM, STRUCTURE_FEAT_DIM, MAX_SEQ, VOCAB, encode_decision
from .engine import play_round
from .model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt


DMC_REWARD_SCHEMA = "terminal_team_score_div3_v1"


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_training(args):
    """Load DMC state or start a new run from weights, never mix their optimizers.

    A BC head produces arbitrary logits. Scaling its last affine layer by a
    positive number preserves action ordering while reducing its initial
    magnitude before fitting normalized team returns. This is initialization,
    not a probability or value calibration; no reward labels are fabricated.
    """
    resume = getattr(args, "resume", "")
    warm_start = getattr(args, "warm_start", "")
    scale = getattr(args, "warm_start_q_scale", None)
    requested_dim = getattr(args, "feature_dim", None)
    if requested_dim is not None and requested_dim not in (FEAT_DIM, STRUCTURE_FEAT_DIM):
        raise ValueError("--feature-dim must be 80 or 224")
    if resume and warm_start:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    if scale is not None and not warm_start:
        raise ValueError("--warm-start-q-scale requires --warm-start")
    if scale is not None and (not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("--warm-start-q-scale must be finite and in (0, 1]")
    source = Path(resume or warm_start).resolve() if (resume or warm_start) else None
    output = Path(args.out).resolve()
    if warm_start and output == source.parent:
        raise ValueError("--warm-start requires a new output directory; preserve its source checkpoint")
    if not resume and any((output / name).exists() for name in
                          ("latest.pt", "best.pt", "latest.npz", "best.npz")):
        raise ValueError("output already contains checkpoints; use --resume or a new --out")

    ck, previous_meta = None, {}
    provenance = {"mode": "scratch"}
    if source is not None:
        ck = torch.load(source, map_location="cpu", weights_only=False)
        meta = ck.get("meta", {})
        kind = meta.get("training_kind")
        if resume:
            legacy_dmc = (kind is None and "cycle" in meta and "total_samples" in meta
                          and "epoch" not in meta and "real_data_manifest_sha256" not in meta)
            if kind != "dmc" and not legacy_dmc:
                raise ValueError("--resume requires a DMC checkpoint; use --warm-start for BC or other weights")
            if meta.get("reward_schema", DMC_REWARD_SCHEMA) != DMC_REWARD_SCHEMA:
                raise ValueError("resume checkpoint uses a different reward schema")
            if meta.get("trainer") and meta["trainer"] != getattr(args, "trainer", meta["trainer"]):
                raise ValueError("--resume must use the same DMC trainer; use --warm-start to switch")
            if not ck.get("optimizer"):
                raise ValueError("DMC resume checkpoint is missing optimizer state; use --warm-start")
            previous_meta = meta
        cfg = ModelConfig.from_dict(ck["config"])
        source_dim, source_version = cfg.feat_dim, cfg.feature_version
        if resume:
            if requested_dim is not None and requested_dim != cfg.feat_dim:
                raise ValueError("--feature-dim differs from resume checkpoint; use --warm-start to migrate")
            if (meta.get("feature_dim", cfg.feat_dim) != cfg.feat_dim
                    or meta.get("feature_version", cfg.feature_version) != cfg.feature_version):
                raise ValueError("resume feature dimension/version metadata disagrees with model config")
        if args.n_blocks is not None and args.n_blocks != cfg.n_blocks:
            raise ValueError("--n-blocks differs from checkpoint architecture")
        if args.ntp_weight is not None:
            if resume and args.ntp_weight != cfg.ntp_weight:
                raise ValueError("--ntp-weight differs from resume checkpoint config")
            cfg.ntp_weight = args.ntp_weight
        provenance = {
            "mode": "resume" if resume else "warm_start",
            "checkpoint": str(source),
            "checkpoint_sha256": checkpoint_sha256(source),
            "source_training_kind": kind or "legacy_dmc",
            "source_real_data_manifest_sha256": meta.get("real_data_manifest_sha256"),
            "source_feature_dim": source_dim, "source_feature_version": source_version,
        }
        if warm_start and requested_dim is not None and requested_dim != source_dim:
            if (source_dim, requested_dim) != (FEAT_DIM, STRUCTURE_FEAT_DIM):
                raise ValueError("only 80-to-224 feature migration is supported at warm start")
            config = cfg.to_dict()
            config.update(feat_dim=STRUCTURE_FEAT_DIM, feature_version=2)
            cfg = ModelConfig.from_dict(config)
            provenance["feature_migration"] = {
                "from_dim": source_dim, "to_dim": cfg.feat_dim,
                "from_version": source_version, "to_version": cfg.feature_version,
                "method": "zero_pad_hand_mlp_first_layer",
                "legacy_scores_preserved_before_q_scale": True,
            }
    else:
        cfg = ModelConfig(n_blocks=args.n_blocks if args.n_blocks is not None else 4,
                          ntp_weight=args.ntp_weight if args.ntp_weight is not None else 0.02,
                          feat_dim=requested_dim if requested_dim is not None else FEAT_DIM)
    model = FableDanNet(cfg).to(args.device)
    if ck is not None:
        state = ck["model"]
        if "feature_migration" in provenance:
            state = dict(state)
            old = state["hand_mlp.0.weight"]
            if old.ndim != 2 or old.shape != (cfg.hand_hidden, FEAT_DIM):
                raise ValueError("source hand MLP input layer disagrees with its 80-feature config")
            padded = old.new_zeros((old.shape[0], cfg.feat_dim))
            padded[:, :FEAT_DIM] = old
            state["hand_mlp.0.weight"] = padded
        model.load_state_dict(state)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("checkpoint contains non-finite model parameters")
    if warm_start:
        if scale is None:
            scale = 0.05 if ck.get("meta", {}).get("training_kind") == "real_bc" else 1.0
        head = model.q_head[-1]
        if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
            raise ValueError("expected a scalar final Q affine layer")
        with torch.no_grad():
            head.weight.mul_(scale)
            head.bias.mul_(scale)
        provenance["q_output_scale"] = scale
        provenance["optimizer_reset"] = True
        provenance["counters_reset"] = True
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or 1e-4)
    if resume:
        opt.load_state_dict(ck["optimizer"])
        if args.lr is not None:
            for group in opt.param_groups:
                group["lr"] = args.lr
    return model, opt, previous_meta, provenance


# ---------------------------------------------------------------------------
# actor
# ---------------------------------------------------------------------------

class _ActorPolicy:
    """eps-greedy/top-k policy over the local model copy."""

    def __init__(self, model, eps, top_k, rng):
        self.model = model
        self.feature_dim = model.cfg.feat_dim
        self.eps = eps
        self.top_k = top_k
        self.rng = rng
        self.samples = []          # (toks, feat_of_chosen, player)

    def act(self, obs):
        toks, feats = encode_decision(obs, feat_dim=self.feature_dim)
        n = feats.shape[0]
        with torch.no_grad():
            t = torch.tensor([toks], dtype=torch.long)
            ln = torch.tensor([len(toks)])
            f = torch.tensor(feats[None], dtype=torch.float32)
            q, _ = self.model(t, ln, f)
            q = q[0].numpy()
        if self.eps > 0 and self.rng.random() < self.eps:
            if self.top_k > 1:
                k = min(self.top_k, n)
                idx = int(self.rng.choice(list(np.argsort(q)[-k:])))
            else:
                idx = self.rng.randrange(n)
        else:
            idx = int(np.argmax(q))
        # Retain only the chosen row, not the full candidate matrix behind a
        # NumPy view (structure candidates make that matrix substantially larger).
        self.samples.append((toks, feats[idx].copy(), obs["player"]))
        return idx


def actor_proc(actor_id, cfg_dict, weight_q, sample_q, stop_ev, seed):
    torch.set_num_threads(1)
    cfg = ModelConfig.from_dict(cfg_dict)
    model = FableDanNet(cfg)
    model.eval()
    rng = random.Random(seed)
    eps = cfg_dict.get("_eps", 0.02)
    top_k = cfg_dict.get("_top_k", 10)
    sd = weight_q.get()                      # initial weights
    model.load_state_dict(sd)
    while not stop_ev.is_set():
        # non-blocking weight refresh
        try:
            while True:
                sd = weight_q.get_nowait()
                model.load_state_dict(sd)
        except Exception:
            pass
        pol = _ActorPolicy(model, eps, top_k, rng)
        agents = [pol, pol, pol, pol]
        rewards, ranking, _ = play_round(agents, rng=random.Random(rng.getrandbits(48)),
                                         feature_dim=cfg.feat_dim)
        out = []
        for toks, feat, player in pol.samples:
            z = rewards[player] / 3.0
            out.append((np.asarray(toks, dtype=np.int16), feat, np.float32(z)))
        sample_q.put(out)


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------

class Replay:
    def __init__(self, capacity, belief_dim=0, feat_dim=FEAT_DIM):
        if capacity <= 0 or belief_dim < 0:
            raise ValueError("replay capacity must be positive and belief_dim nonnegative")
        self.capacity = capacity
        if feat_dim not in (FEAT_DIM, STRUCTURE_FEAT_DIM):
            raise ValueError("unsupported replay feature dimension")
        self.feat_dim = int(feat_dim)
        self.feature_version = 1 if self.feat_dim == FEAT_DIM else 2
        self.toks = [None] * capacity
        self.feat = np.zeros((capacity, self.feat_dim), dtype=np.float32)
        self.targ = np.zeros(capacity, dtype=np.float32)
        self.belief_dim = belief_dim
        if belief_dim:
            self.belief = np.zeros((capacity, belief_dim), dtype=np.float32)
        self.n = 0
        self.ptr = 0

    def add(self, toks, feat, targ, belief=None):
        toks = np.asarray(toks)
        feat = np.asarray(feat, dtype=np.float32)
        if (toks.ndim != 1 or not 0 < len(toks) <= MAX_SEQ
                or not np.issubdtype(toks.dtype, np.integer)
                or np.any(toks < 0) or np.any(toks >= VOCAB)):
            raise ValueError("invalid replay token sequence")
        if feat.shape != (self.feat_dim,) or not np.all(np.isfinite(feat)):
            raise ValueError("invalid replay action features")
        if not np.isfinite(targ) or abs(targ) > 1.000001:
            raise ValueError("replay target must be normalized terminal team return in [-1, 1]")
        if self.belief_dim:
            belief = np.asarray(belief, dtype=np.float32)
            if belief.shape != (self.belief_dim,) or not np.all(np.isfinite(belief)):
                raise ValueError("missing or invalid replay belief label")
        i = self.ptr
        self.toks[i] = toks.astype(np.int16, copy=True)
        self.feat[i] = feat
        self.targ[i] = targ
        if self.belief_dim and belief is not None:
            self.belief[i] = belief
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, bs, rng):
        if self.n == 0 or bs <= 0:
            raise ValueError("replay sampling requires data and a positive batch size")
        idx = rng.integers(0, self.n, size=bs)
        toks = [self.toks[i] for i in idx]
        maxlen = max(len(t) for t in toks)
        T = np.zeros((bs, maxlen), dtype=np.int64)
        L = np.zeros(bs, dtype=np.int64)
        for j, t in enumerate(toks):
            T[j, :len(t)] = t
            L[j] = len(t)
        B = self.belief[idx] if self.belief_dim else None
        return T, L, self.feat[idx], self.targ[idx], B

    def state_dict(self):
        """Persist physical FIFO slots during synchronous checkpoint writing.

        Array views avoid a second full replay copy. The learner must finish
        saving before adding samples again (both training entrypoints do so).
        """
        return {
            "schema": "fabledan_replay_v2", "capacity": self.capacity,
            "feat_dim": self.feat_dim, "feature_version": self.feature_version,
            "belief_dim": self.belief_dim, "n": self.n, "ptr": self.ptr,
            "tokens": self.toks[:self.n],
            "features": self.feat[:self.n], "targets": self.targ[:self.n],
            "belief": self.belief[:self.n] if self.belief_dim else None,
        }

    def load_state_dict(self, state):
        if (state.get("schema") not in ("fabledan_replay_v1", "fabledan_replay_v2")
                or state.get("capacity") != self.capacity
                or state.get("belief_dim") != self.belief_dim):
            raise ValueError("resume replay schema/capacity/belief_dim differs; keep --buffer and --belief-weight mode")
        if (state.get("feat_dim", FEAT_DIM) != self.feat_dim
                or state.get("feature_version", 1) != self.feature_version):
            raise ValueError("resume replay feature dimension/version differs from model")
        n, ptr = state["n"], state["ptr"]
        if not 0 <= n <= self.capacity or not 0 <= ptr < self.capacity or (n < self.capacity and ptr != n):
            raise ValueError("invalid replay cursor")
        if (len(state["tokens"]) != n or np.shape(state["features"]) != (n, self.feat_dim)
                or np.shape(state["targets"]) != (n,)
                or (self.belief_dim and np.shape(state["belief"]) != (n, self.belief_dim))):
            raise ValueError("invalid replay checkpoint shapes")
        restored = Replay(self.capacity, self.belief_dim, self.feat_dim)
        for i in range(n):
            restored.add(state["tokens"][i], state["features"][i], state["targets"][i],
                         state["belief"][i] if self.belief_dim else None)
        self.toks, self.feat, self.targ = restored.toks, restored.feat, restored.targ
        if self.belief_dim:
            self.belief = restored.belief
        self.n, self.ptr = n, ptr


# ---------------------------------------------------------------------------
# learner
# ---------------------------------------------------------------------------

def quick_eval(model, games=100, seed=123):
    """Greedy model (team A) vs RuleAgent. Returns win rate."""
    from .agents import RuleAgent, TorchAgent
    from .evaluate import evaluate
    model_cpu = FableDanNet(model.cfg)
    model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    wr, _ = evaluate(lambda: TorchAgent(model_cpu), lambda: RuleAgent(),
                     games=games, seed=seed)
    return wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ckpts/run1")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--resume", default="", help="continue a DMC run including optimizer and replay")
    start.add_argument("--warm-start", default="", help="new DMC run from BC or compatible model weights")
    ap.add_argument("--warm-start-q-scale", type=float, default=None,
                    help="positive final-Q scale at warm start (BC: 0.05; other: 1)")
    ap.add_argument("--feature-dim", type=int, choices=(FEAT_DIM, STRUCTURE_FEAT_DIM), default=None,
                    help="feature dimension (default: source checkpoint or 80); migrate only via warm start")
    ap.add_argument("--actors", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--cycles", type=int, default=1000000)
    ap.add_argument("--buffer", type=int, default=131072)
    ap.add_argument("--diversity", type=int, default=2)
    ap.add_argument("--steps-per-cycle", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--ntp-weight", type=float, default=None)
    ap.add_argument("--eval-cycles", type=int, default=20)
    ap.add_argument("--ckpt-cycles", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    args.trainer = "train"
    for name in ("actors", "cycles", "buffer", "diversity", "steps_per_cycle", "batch",
                 "top_k", "eval_cycles", "ckpt_cycles"):
        if getattr(args, name) <= 0:
            ap.error("--%s must be positive" % name.replace("_", "-"))
    if args.buffer < args.diversity or not 0 <= args.eps <= 1 or args.seed < 0:
        ap.error("require buffer >= diversity, eps in [0,1], and nonnegative seed")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        ap.error("--lr must be finite and positive")
    if args.ntp_weight is not None and (not math.isfinite(args.ntp_weight) or args.ntp_weight < 0):
        ap.error("--ntp-weight must be finite and nonnegative")
    device = args.device
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    try:
        model, opt, previous_meta, startup = initialize_training(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))
    cfg = model.cfg
    start_cycle = previous_meta.get("cycle", 0)
    total_samples = previous_meta.get("total_samples", 0)
    if args.resume:
        print(f"resumed from {args.resume} at cycle {start_cycle}")
    if args.cycles <= start_cycle:
        ap.error("--cycles is the total target; it must exceed resumed cycle")
    replay = Replay(args.buffer, feat_dim=cfg.feat_dim)
    if "replay" in previous_meta:
        try:
            replay.load_state_dict(previous_meta["replay"])
        except (ValueError, KeyError) as exc:
            ap.error(str(exc))
        previous_meta = dict(previous_meta)
        previous_meta.pop("replay")
    elif args.resume:
        print("legacy checkpoint has no replay; collecting fresh samples before learning", flush=True)
    rng = np.random.default_rng(args.seed)
    if "sample_rng" in previous_meta:
        rng.bit_generator.state = previous_meta["sample_rng"]
    if "torch_rng" in previous_meta:
        torch.set_rng_state(previous_meta["torch_rng"].cpu())
    if torch.cuda.is_available() and "cuda_rng" in previous_meta:
        torch.cuda.set_rng_state_all([state.cpu() for state in previous_meta["cuda_rng"]])
    os.makedirs(args.out, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    weight_qs = [mp.Queue(maxsize=4) for _ in range(args.actors)]
    sample_q = mp.Queue(maxsize=256)
    stop_ev = mp.Event()
    cfg_dict = cfg.to_dict()
    cfg_dict["_eps"] = args.eps
    cfg_dict["_top_k"] = args.top_k
    procs = []
    def broadcast():
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for q in weight_qs:
            try:
                q.put_nowait(sd)
            except pyqueue.Full:
                pass
    cycle_budget = args.buffer // args.diversity
    best_wr = previous_meta.get("best_wr", 0.0)
    optimizer_steps = previous_meta.get("optimizer_steps", 0)
    completed_cycle = start_cycle
    initial_samples = total_samples
    t_start = time.monotonic()
    stop_reason = "cycles completed"
    # Imported at runtime because train_fast imports Replay and initialization
    # from this module; reuse the same atomic writes and bounded shutdown.
    from .train_fast import atomic_checkpoint, atomic_export, check_workers, shutdown_workers

    def metadata():
        meta = {"training_kind": "dmc", "trainer": "train", "checkpoint_version": 2,
                "feature_dim": cfg.feat_dim, "feature_version": cfg.feature_version,
                "reward_schema": DMC_REWARD_SCHEMA,
                "objective": "MSE to completed-round team score / 3 plus auxiliary NTP",
                "initialization": previous_meta.get("initialization", startup), "startup": startup,
                "cycle": completed_cycle, "total_samples": total_samples, "best_wr": best_wr,
                "optimizer_steps": optimizer_steps, "replay": replay.state_dict(),
                "sample_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "training_args": vars(args).copy(), "stop_reason": stop_reason,
                "resume_scope": "model, optimizer, replay and learner RNG; actors restart new rounds"}
        if torch.cuda.is_available():
            meta["cuda_rng"] = torch.cuda.get_rng_state_all()
        return meta

    try:
        for a in range(args.actors):
            p = mp.Process(target=actor_proc, name="actor-%d" % a,
                           args=(a, cfg_dict, weight_qs[a], sample_q, stop_ev,
                                 args.seed + a + start_cycle * args.actors), daemon=True)
            p.start()
            procs.append(p)
        broadcast()
        for cycle in range(start_cycle, args.cycles):
            got = 0
            t0 = time.monotonic()
            while got < cycle_budget:
                check_workers(procs)
                try:
                    ep = sample_q.get(timeout=1)
                except pyqueue.Empty:
                    continue
                for toks, feat, z in ep:
                    replay.add(toks, feat, z)
                got += len(ep)
                total_samples += len(ep)
            t_collect = time.monotonic() - t0
            t0 = time.monotonic()
            model.train()
            losses = []
            for _ in range(args.steps_per_cycle):
                T, L, F, Z, _B = replay.sample(args.batch, rng)
                T = torch.from_numpy(T).to(device)
                L = torch.from_numpy(L).to(device)
                F = torch.from_numpy(F).to(device).unsqueeze(1)
                Z = torch.from_numpy(Z).to(device)
                q, hid = model(T, L, F)
                loss_q = torch.nn.functional.mse_loss(q[:, 0], Z)
                loss = loss_q
                if cfg.ntp_weight > 0:
                    loss = loss + cfg.ntp_weight * model.ntp_loss(T, hid)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("non-finite DMC loss; refusing optimizer update")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                optimizer_steps += 1
                losses.append(loss_q.item())
            model.eval()
            completed_cycle = cycle + 1
            t_train = time.monotonic() - t0
            broadcast()
            sps = (total_samples - initial_samples) / (time.monotonic() - t_start + 1e-9)
            print(f"cycle {completed_cycle} samples {total_samples} loss {np.mean(losses):.4f} "
                  f"collect {t_collect:.1f}s train {t_train:.1f}s {sps:.0f} samples/s", flush=True)
            if completed_cycle % args.eval_cycles == 0:
                wr = quick_eval(model, games=60)
                print(f"  eval vs rule: {wr:.1%}", flush=True)
                if wr >= best_wr:
                    best_wr = wr
                    atomic_checkpoint(model, opt, metadata(), os.path.join(args.out, "best.pt"))
                    atomic_export(model, os.path.join(args.out, "best.npz"))
            if completed_cycle % args.ckpt_cycles == 0:
                atomic_checkpoint(model, opt, metadata(), os.path.join(args.out, "latest.pt"))
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception:
        stop_reason = "training failed"
        raise
    finally:
        try:
            shutdown_workers(stop_ev, procs, weight_qs + [sample_q])
        finally:
            model.eval()
            atomic_checkpoint(model, opt, metadata(), os.path.join(args.out, "latest.pt"))
            atomic_export(model, os.path.join(args.out, "latest.npz"))


if __name__ == "__main__":
    main()
