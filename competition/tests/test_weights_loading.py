"""A packed Bot must use the model version recorded in its code archive."""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "botzone"))

import bot_fabledan as B


def mlp_blob(path, marker=0):
    from fabledan.encode import FLAT_DIM
    np.savez(path, W0=np.zeros((FLAT_DIM, 1), dtype=np.float32),
             W1=np.zeros((1, 1), dtype=np.float32),
             W2=np.zeros((1, 1), dtype=np.float32),
             b0=np.zeros(1, dtype=np.float32),
             b1=np.zeros(1, dtype=np.float32),
             b2=np.array([marker], dtype=np.float32))
    return path.read_bytes()


def load_from(folder):
    # Give the loader the same layout as an extracted submission, without
    # relying on any real weights or user storage in the workspace.
    with patch.object(B, "__file__", str(folder / "__main__.py")):
        old = os.getcwd()
        os.chdir(str(folder))
        try:
            return B._load_model(), B.WEIGHTS_INFO
        finally:
            os.chdir(old)


def test_pinned_weights_do_not_load_a_different_version():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / "data").mkdir()
        legacy = folder / "data" / "fabledan_weights.npz"
        blob = mlp_blob(legacy, marker=17)
        name = "fabledan_w_%s.npz" % hashlib.sha256(blob).hexdigest()[:8]
        (folder / "weights_name.txt").write_text(name + "\n", encoding="utf-8")
        (model, info) = load_from(folder)
        assert model[0] == "rule" and info == "missing(%s)" % name

        pinned = folder / "data" / name
        pinned.write_bytes(b"corrupt archive")
        (model, info) = load_from(folder)
        assert model[0] == "rule" and info == "invalid(%s)" % name

        # A valid archive renamed to the pinned filename is still the wrong
        # model; this was a silent version mismatch before SHA verification.
        mlp_blob(pinned, marker=23)
        (model, info) = load_from(folder)
        assert model[0] == "rule" and info == "invalid(%s)" % name

        pinned.write_bytes(blob)
        (model, info) = load_from(folder)
        assert model[0] == "mlp" and model[1].b[2][0] == 17
        assert info == os.path.join("data", name)


def test_invalid_pin_is_explicit_and_unpinned_legacy_still_loads():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        mlp_blob(folder / "fabledan_weights.npz", marker=31)
        pin = folder / "weights_name.txt"
        for invalid in ("", "../wrong.npz", "data/wrong.npz"):
            pin.write_text(invalid, encoding="utf-8")
            (model, info) = load_from(folder)
            assert model[0] == "rule" and info == "invalid(weights_name.txt)"
        pin.unlink()
        (model, info) = load_from(folder)
        assert model[0] == "mlp" and model[1].b[2][0] == 31
        assert info.endswith("fabledan_weights.npz")


if __name__ == "__main__":
    test_pinned_weights_do_not_load_a_different_version()
    test_invalid_pin_is_explicit_and_unpinned_legacy_still_loads()
    print("pinned weights and explicit fallback contracts OK")
