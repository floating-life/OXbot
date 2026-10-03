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
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOT_MODULES = ["__init__.py", "cards.py", "combos.py", "engine.py",
               "encode.py", "model_np.py", "agents.py", "train_demo.py"]


def weights_name_for(npz_path):
    h = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "fabledan_w_%s.npz" % h.hexdigest()[:8]


def pack(weights_npz=None, out_zip=None, embed=False):
    """-> (zip_path, weights_copy_path_or_None, weights_name_or_None)"""
    out_zip = out_zip or os.path.join(ROOT, "dist", "fabledan_bot.zip")
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    wname = weights_name_for(weights_npz) if weights_npz else None
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(ROOT, "botzone", "bot_fabledan.py"), "__main__.py")
        for m in BOT_MODULES:
            z.write(os.path.join(ROOT, "fabledan", m), "fabledan/" + m)
        if weights_npz and embed:
            z.write(weights_npz, "fabledan_weights.npz")
        elif wname:
            z.writestr("weights_name.txt", wname + "\n")
    wcopy = None
    if weights_npz and not embed:
        wcopy = os.path.join(os.path.dirname(out_zip), wname)
        if os.path.abspath(wcopy) != os.path.abspath(weights_npz):
            shutil.copyfile(weights_npz, wcopy)
    return out_zip, wcopy, wname
