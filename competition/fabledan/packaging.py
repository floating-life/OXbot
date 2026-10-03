# -*- coding: utf-8 -*-
"""Build the Botzone upload: code zip + versioned weights file.

The default transformer is ~16 MB of weights, far above Botzone's 4 MB
source limit, so weights go to the user storage space (用户存储空间):

  dist/<tag>.zip                 upload as the bot source (python3 / Python 3.6)
  dist/fabledan_w_<sha8>.npz     upload to 用户存储空间 (keep the file name!)

The zip carries weights_name.txt -> the bot loads data/fabledan_w_<sha8>.npz,
so several weight versions can live in storage side by side and every bot
version keeps using the weights it was packed with.
"""

import hashlib
import os
import shutil
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOT_MODULES = ["__init__.py", "cards.py", "combos.py", "engine.py",
               "encode.py", "model_np.py", "agents.py", "train_demo.py"]
SOURCE_LIMIT = 4000000


def weights_name_for(npz_path):
    h = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "fabledan_w_%s.npz" % h.hexdigest()[:8]


def pack(weights_npz=None, out_zip=None, embed=False):
    """-> (zip_path, weights_copy_path_or_None, weights_name_or_None)"""
    out_zip = out_zip or os.path.join(ROOT, "dist", "fabledan_bot.zip")
    out_zip = os.path.abspath(out_zip)
    out_dir = os.path.dirname(out_zip)
    os.makedirs(out_dir, exist_ok=True)
    wname = weights_name_for(weights_npz) if weights_npz else None
    if weights_npz:
        # Reject malformed/non-model files before producing an upload artifact.
        import numpy as np
        from .model_np import NumpyModel
        with np.load(weights_npz, allow_pickle=False) as arrays:
            if "token_emb.weight" not in arrays.files:
                raise ValueError("pack requires exported FableDan transformer weights")
            model = NumpyModel({k: arrays[k] for k in arrays.files})
        scores = model.q_values([1], np.zeros((1, model.feat_dim), np.float32))
        if scores.shape != (1,) or not np.isfinite(scores).all():
            raise ValueError("weights produced invalid Q values")
    fd, tmp_zip = tempfile.mkstemp(suffix=".zip", dir=out_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(ROOT, "botzone", "bot_fabledan.py"), "__main__.py")
            for m in BOT_MODULES:
                z.write(os.path.join(ROOT, "fabledan", m), "fabledan/" + m)
            if weights_npz and embed:
                z.write(weights_npz, "fabledan_weights.npz")
            elif wname:
                z.writestr("weights_name.txt", wname + "\n")
        if os.path.getsize(tmp_zip) >= SOURCE_LIMIT:
            raise ValueError("Botzone source zip must be below 4,000,000 bytes; "
                             "use separate user-storage weights")
        os.replace(tmp_zip, out_zip)
    finally:
        if os.path.exists(tmp_zip):
            os.unlink(tmp_zip)
    wcopy = None
    if weights_npz and not embed:
        wcopy = os.path.join(out_dir, wname)
        if os.path.abspath(wcopy) != os.path.abspath(weights_npz):
            shutil.copyfile(weights_npz, wcopy)
    return out_zip, wcopy, wname
