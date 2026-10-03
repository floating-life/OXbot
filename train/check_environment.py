"""Verify GPU execution, backward pass, BF16 and the actual deployment model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time
import torch

from model import CandidateModel, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/training_environment.json"))
    parser.add_argument("--lock", type=Path, default=Path("train/requirements.lock.txt"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    torch.manual_seed(20261001)
    torch.cuda.reset_peak_memory_stats()
    c = ModelConfig()
    model = CandidateModel(c).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tokens = torch.randint(1, c.vocabulary, (32, c.history_length), device="cuda")
    lengths = torch.full((32,), c.history_length, device="cuda")
    state = torch.randn(32, c.state_features, device="cuda")
    actions = torch.randn(32, 128, c.action_features, device="cuda")
    target = torch.arange(32, device="cuda")
    supports_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if supports_bf16 else torch.float32
    started = time.monotonic()
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=supports_bf16):
            logits = model(tokens, lengths, state, actions)
            loss = torch.nn.functional.cross_entropy(logits, target)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    assert losses[-1] < losses[0], losses
    report = {"status": "passed", "python": platform.python_version(), "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "capability": torch.cuda.get_device_capability(0), "bf16": supports_bf16,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
              "four_training_steps_seconds": time.monotonic() - started,
              "synthetic_losses": losses, "architecture": model.architecture(),
              "note": "Synthetic environment test only; no gameplay model has been trained by this command."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.lock.write_text(subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"],
                                               text=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
