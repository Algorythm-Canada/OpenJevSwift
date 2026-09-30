#!/bin/bash
# Measure every converted package with every compute-unit setting on this Mac, one process each.
#
#   Tools/encoders/run_macos.sh [package ...]
#
# Without arguments it measures every package that Tools/encoders/convert_*.py wrote. Each result
# goes to docs/spikes/encoder-runtime/macos/<package>-<units>.json. A configuration whose process
# exits abnormally (Core ML's CPU backend traps on some packages; see docs/spikes/encoder-runtime.md)
# is recorded in docs/spikes/encoder-runtime/macos/failures.txt with its exit status and the last
# lines it printed, and the run goes on with the next configuration. A run of every package starts
# failures.txt afresh; a run of named packages appends to it.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HARNESS="$ROOT/Tools/encoders/Harness"
OUT="$ROOT/docs/spikes/encoder-runtime/macos"
MODELS="${OPENJEV_ENCODER_MODELS:-$HOME/Library/Caches/OpenJevSwift/encoders}"
PACKAGES=("$@")
FRESH=0
if [ ${#PACKAGES[@]} -eq 0 ]; then
    PACKAGES=(verdict-e17-fp16 verdict-e17-fp32 verdict-m18-fp16 laya-e17-fp16 laya-m18-fp16)
    FRESH=1
fi

swift build --package-path "$HARNESS" -c release --product encoder-harness || exit 1
BIN="$(swift build --package-path "$HARNESS" -c release --show-bin-path)/encoder-harness"
mkdir -p "$OUT"
if [ $FRESH -eq 1 ]; then
    : > "$OUT/failures.txt"
fi
for package in "${PACKAGES[@]}"; do
    [ -d "$MODELS/$package.mlpackage" ] || { echo "skip $package: not converted"; continue; }
    for units in cpuAndGPU all cpuOnly cpuAndNeuralEngine; do
        log="$OUT/$package-$units.log"
        "$BIN" --package "$package" --units "$units" --root "$ROOT" --output "$OUT/$package-$units.json" > "$log" 2>&1
        status=$?
        if [ $status -ne 0 ]; then
            rm -f "$OUT/$package-$units.json"
            { echo "$package $units: exit status $status"; tail -3 "$log" | sed 's/^/    /'; } >> "$OUT/failures.txt"
            echo "$package $units: exit status $status"
        else
            echo "$package $units: done"
        fi
        rm -f "$log"
    done
done
