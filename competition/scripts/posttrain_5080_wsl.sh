#!/usr/bin/env bash
set -Eeuo pipefail
competition_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
oxbot_python="${OXBOT_PYTHON:-/home/ggcle/.venvs/oxbot/bin/python}"
cd "$competition_root"
exec "$oxbot_python" posttrain.py "$@"
