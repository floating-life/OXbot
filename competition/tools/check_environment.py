# -*- coding: utf-8 -*-
"""Verify the real FableDan CUDA learner at the configured maximum sequence.

This uses synthetic data to check finite loss, backward, optimizer update and
PyTorch/NumPy exported-weight parity; it does not measure playing strength.
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fabledan.model_torch import FableDanNet, ModelConfig, export_npz
from fabledan.model_np import NumpyModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-batch", type=int, default=256)
    parser.add_argument("--report", default="reports/training_environment.json")
    args = parser.parse_args()
    if args.micro_batch <= 0:
        parser.error("micro-batch must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: install a sm_120-compatible PyTorch cu128+ build")
    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    torch.cuda.reset_peak_memory_stats()
    config = ModelConfig()
    model = FableDanNet(config).to("cuda:0")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tokens = torch.randint(1, config.vocab, (args.micro_batch, config.max_seq), device="cuda:0")
    lengths = torch.full((args.micro_batch,), config.max_seq, device="cuda:0")
    features = torch.randn(args.micro_batch, 1, config.feat_dim, device="cuda:0")
    target = torch.randn(args.micro_batch, device="cuda:0")
    before = model.q_head[-1].weight.detach().clone()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q, hid = model(tokens, lengths, features)
        loss = torch.nn.functional.mse_loss(q[:, 0].float(), target)
        loss = loss + config.ntp_weight * model.ntp_loss(tokens, hid)
        loss = loss + .05 * model.belief_loss(hid[:, -1].float(),
                                             torch.zeros(args.micro_batch, 45, device="cuda:0"))
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite learner loss")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("nonfinite learner gradients")
    opt.step()
    torch.cuda.synchronize()
    if torch.equal(before, model.q_head[-1].weight):
        raise RuntimeError("optimizer did not update the model")
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    model.eval()
    with torch.no_grad():
        expected = model(tokens[:1, :16], lengths[:1].clamp(max=16), features[:1])[0].float().cpu().numpy()[0]
    with tempfile.TemporaryDirectory(prefix="oxbot_parity_") as directory:
        weight_path = os.path.join(directory, "test.npz")
        export_npz(model, weight_path)
        numpy_model = NumpyModel(weight_path)
        actual = numpy_model.q_values(tokens[0, :16].cpu().tolist(), features[0].cpu().numpy())
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-4):
        raise RuntimeError("exported NumPy scores differ: %s" % error)
    report = {"status": "passed", "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0),
              "capability": torch.cuda.get_device_capability(0),
              "architecture": config.to_dict(), "micro_batch": args.micro_batch,
              "max_sequence": config.max_seq, "loss": float(loss.detach()),
              "peak_allocated_mib": peak_mib, "update_seconds": elapsed,
              "numpy_parity_max_error": error,
              "note": "synthetic learner check; self-play tested separately"}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
