#!/bin/bash
# Fill the harness app's Staged folder for a run on an iPhone, which cannot read the Mac's files.
#
#   Tools/encoders/stage_harness.sh [package ...]   fixtures, both tokenizers and the named packages
#   Tools/encoders/stage_harness.sh --clean         empty it again (only the README stays)
#
# Packages are the names Tools/encoders/convert_*.py write, for example verdict-m18-fp16. They are
# read from OPENJEV_ENCODER_MODELS, by default ~/Library/Caches/OpenJevSwift/encoders. The
# tokenizers are read from the Hugging Face cache at the revisions Fixtures/encoders records.
# Git ignores everything staged except the README.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STAGED="$ROOT/Tools/encoders/HarnessApp.swiftpm/App/Staged"
MODELS="${OPENJEV_ENCODER_MODELS:-$HOME/Library/Caches/OpenJevSwift/encoders}"
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

find "$STAGED" -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +
if [ "${1:-}" = "--clean" ]; then
    echo "emptied $STAGED"
    exit 0
fi

revision() {
    /usr/bin/python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["generator"][sys.argv[2]])' \
        "$ROOT/Fixtures/encoders/verdict.json" "$1"
}
VERDICT="$HUB/models--heman10x--rlcd-modernbert-151m/snapshots/$(revision verdict_revision)"
LAYA="$HUB/models--convaiinnovations--laya-typed-decisions/snapshots/$(revision laya_revision)/tokenizer"

mkdir -p "$STAGED/fixtures" "$STAGED/tokenizers/verdict" "$STAGED/tokenizers/laya" "$STAGED/models"
cp "$ROOT/Fixtures/encoders/verdict.json" "$ROOT/Fixtures/encoders/laya.json" "$STAGED/fixtures/"
# The Hugging Face cache holds symbolic links into its blob store; copy the files they point to.
cp -L "$VERDICT/tokenizer.json" "$VERDICT/tokenizer_config.json" "$STAGED/tokenizers/verdict/"
cp -L "$LAYA/tokenizer.json" "$LAYA/tokenizer_config.json" "$STAGED/tokenizers/laya/"
for package in "$@"; do
    if [ ! -d "$MODELS/$package.mlpackage" ]; then
        echo "no $MODELS/$package.mlpackage; run Tools/encoders/convert_verdict.py or convert_laya.py" >&2
        exit 1
    fi
    cp -R "$MODELS/$package.mlpackage" "$STAGED/models/"
done
du -sh "$STAGED"
