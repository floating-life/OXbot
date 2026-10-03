#!/usr/bin/env bash
set -euo pipefail

oxbot_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$oxbot_root/bin"
oxbot_cxx="${CXX:-g++}"
oxbot_flags=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -I"$oxbot_root/core/include")
SOURCES=(
  "$oxbot_root/core/src/rules.cpp"
  "$oxbot_root/core/src/state.cpp"
  "$oxbot_root/core/src/policy.cpp"
  "$oxbot_root/core/src/protocol.cpp"
  "$oxbot_root/core/src/network.cpp"
  "$oxbot_root/core/src/fabledan_network.cpp"
  "$oxbot_root/core/src/features.cpp"
  "$oxbot_root/core/src/fabledan_features.cpp"
  "$oxbot_root/core/src/fabledan_candidates.cpp"
)

"$oxbot_cxx" "${oxbot_flags[@]}" "${SOURCES[@]}" "$oxbot_root/botzone/main.cpp" -o "$oxbot_root/bin/oxbot"
"$oxbot_cxx" "${oxbot_flags[@]}" "${SOURCES[@]}" "$oxbot_root/tools/core_probe.cpp" -o "$oxbot_root/bin/core_probe"
"$oxbot_cxx" "${oxbot_flags[@]}" "$oxbot_root/core/src/network.cpp" "$oxbot_root/tools/network_probe.cpp" -o "$oxbot_root/bin/network_probe"
"$oxbot_cxx" "${oxbot_flags[@]}" "$oxbot_root/core/src/network.cpp" "$oxbot_root/core/src/fabledan_network.cpp" "$oxbot_root/tools/fabledan_probe.cpp" -o "$oxbot_root/bin/fabledan_probe"
for oxbot_test in smoke json_test rules_test state_test; do
  if [[ -f "$oxbot_root/tests/$oxbot_test.cpp" ]]; then
    "$oxbot_cxx" "${oxbot_flags[@]}" "${SOURCES[@]}" "$oxbot_root/tests/$oxbot_test.cpp" -o "$oxbot_root/bin/$oxbot_test"
    "$oxbot_root/bin/$oxbot_test"
  fi
done
printf 'built %s\n' "$oxbot_root/bin/oxbot"
