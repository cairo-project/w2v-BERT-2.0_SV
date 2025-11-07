#!/usr/bin/env bash
# Simple helper to export the W2V-BERT speaker model (preprocessed interface) reproducibly.
# Usage:
#   ./scripts/run_export_preprocessed.sh [CHECKPOINT] [MODEL_DIR] [OUTPUT]
# Example:
#   ./scripts/run_export_preprocessed.sh ../pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model_lmft_0.14.pth ../pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Virtualenv to use (recommended)
VENV_REL=".venv_w2vbert_notebook"
VENV_PATH="$REPO_ROOT/$VENV_REL"
ACTIVATE="$VENV_PATH/bin/activate"

if [ ! -f "$ACTIVATE" ]; then
  echo "Virtualenv activate script not found at $ACTIVATE"
  echo "Create or adjust the venv path, then re-run. Example to create:"
  echo "  python -m venv $VENV_PATH && source $ACTIVATE && pip install -U pip"
  exit 1
fi

# Resolve inputs (allow CLI overrides)
CHECKPOINT="${1:-$REPO_ROOT/../pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model_lmft_0.14.pth}"
MODEL_DIR="${2:-$REPO_ROOT/../pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0}"
OUTPUT="${3:-$REPO_ROOT/packages/w2vbert_speaker/artifacts/w2vbert_speaker_script.pt}"

echo "Repository root: $REPO_ROOT"
echo "Using venv: $VENV_PATH"
echo "Checkpoint: $CHECKPOINT"
echo "Model dir: $MODEL_DIR"
echo "Output artifact: $OUTPUT"

# Activate virtualenv
# shellcheck source=/dev/null
source "$ACTIVATE"

# Ensure package is installed into the environment (editable during development)
echo "Installing package (editable) into active environment..."
pip install -e "$REPO_ROOT/packages/w2vbert_speaker"

# Create artifacts dir
mkdir -p "$(dirname "$OUTPUT")"

# Run export script with preprocessed mode (preferred for parity with eager model)
python3 "$REPO_ROOT/scripts/export_w2vbert_torchscript.py" \
  --checkpoint "$CHECKPOINT" \
  --model-path "$MODEL_DIR" \
  --output "$OUTPUT" \
  --preprocess \
  --device cpu

echo "Export finished. Artifact(s) written to: $(dirname "$OUTPUT")"
