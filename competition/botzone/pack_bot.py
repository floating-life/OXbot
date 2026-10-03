# -*- coding: utf-8 -*-
"""Pack the Botzone submission.

    python botzone/pack_bot.py --weights ckpts/run1/latest.npz
      -> dist/fabledan_bot.zip          upload as bot source (Python 3.6.5)
      -> dist/fabledan_w_<sha8>.npz     upload to 用户存储空间, keep the name

Weights are ~16 MB (> 4 MB source limit), so they live in user storage; the
zip records which file to load (weights_name.txt).  --embed-weights puts the
weights inside the zip instead (only for small custom models).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fabledan.packaging import pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--embed-weights", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "dist",
                                                  "fabledan_bot.zip"))
    args = ap.parse_args()
    zp, wcopy, wname = pack(args.weights or None, args.out, args.embed_weights)
    size = os.path.getsize(zp)
    print("code zip : %s (%.1f KB)" % (zp, size / 1024.0))
    if wcopy:
        print("weights  : %s  -> upload to 用户存储空间 with exactly this name"
              % wcopy)
    elif not args.weights:
        print("no weights: the bot will fall back to the rule policy")


if __name__ == "__main__":
    main()
