#!/usr/bin/env bash
# Create a reproducible virtual environment for notebook work and register an IPython kernel.
# Usage:
#   ./scripts/setup_notebook_env.sh [PYTHON_EXECUTABLE] [VENV_DIR]
# Examples:
#   ./scripts/setup_notebook_env.sh            # uses `python3` and .venv_w2vbert_notebook
#   ./scripts/setup_notebook_env.sh /usr/bin/python3.11 .venv_w2vbert_notebook

set -euo pipefail

REQUESTED_PYTHON=${1:-python3}
VENV_DIR=${2:-.venv_w2vbert_notebook}
KERNEL_NAME=${3:-w2vbert_notebook}
KERNEL_DISPLAY_NAME=${4:-"W2V-BERT Notebook (.venv_w2vbert_notebook)"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$REPO_ROOT/$VENV_DIR"

# Helper: return "major.minor" of a python executable or empty on failure
_py_version() {
  local exe="$1"
  "$exe" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo ""
}

# Find a compatible python (>=3.10, <3.14). Preference order: requested, common patch releases.
choose_python() {
  local try_list=("$REQUESTED_PYTHON" "python3.12" "python3.11" "python3.10" "python3")
  for p in "${try_list[@]}"; do
    if command -v "$p" >/dev/null 2>&1; then
      local ver
      ver=$(_py_version "$p")
      if [ -n "$ver" ]; then
        IFS='.' read -r maj min <<<"$ver"
        if [ "$maj" -eq 3 ] && [ "$min" -ge 10 ] && [ "$min" -le 13 ]; then
          echo "$p"
          return 0
        fi
      fi
    fi
  done
  return 1
}

# (PYTHON_BIN is selected below; create and activate the venv after selection)

# Install core runtime dependencies (package also declares most deps)
# Install numpy pinned to <2 to avoid PyTorch binary incompatibilities with NumPy 2.x.
# See https://numpy.org/dev/release/2.0.0-notes.html and PyTorch release notes for details.
SKIP_AUDIO_PACKAGES=0
PYTHON_BIN="$(choose_python)"
if [ -z "$PYTHON_BIN" ]; then
  # No compatible system python found. If the user supplied a REQUESTED_PYTHON
  # executable, we can proceed with it but must skip audio packages that
  # currently require older Python (numba via librosa). Otherwise fail fast.
  if command -v "$REQUESTED_PYTHON" >/dev/null 2>&1; then
    PYTHON_BIN="$REQUESTED_PYTHON"
    SKIP_AUDIO_PACKAGES=1
    cat <<'MSG' >&2
WARNING: No compatible Python interpreter (>=3.10,<3.14) was found on PATH.
Proceeding with the requested Python anyway, but some audio-related packages
(librosa, torchaudio) will be skipped because they currently require Python <3.14
or have compiled dependencies not available for your Python version.

If you need full audio support, install Python 3.11 or 3.12 and re-run:
  ./scripts/setup_notebook_env.sh /path/to/python3.11

Detected attempted python: ${REQUESTED_PYTHON}
MSG
  else
    cat <<'MSG' >&2
ERROR: No compatible Python interpreter found.
This project requires Python >=3.10 and <3.14 because some dependencies (numba / compiled wheels)
are not yet compatible with Python 3.14+.

Options:
  * Install Python 3.11 or 3.12 and re-run:
      ./scripts/setup_notebook_env.sh /path/to/python3.11
  * Use pyenv / conda to create a compatible interpreter, then pass its path above.
  * If you intentionally want to use Python 3.14+, you must ensure all native dependencies have wheels
    built for 3.14; this is not currently recommended.

Detected attempted python: ${REQUESTED_PYTHON}
MSG
    exit 1
  fi
fi

# Create virtualenv if missing (use the chosen python interpreter)
if [ ! -d "$VENV_PATH" ]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

# Activate the venv
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

# Upgrade pip/setuptools/wheel inside venv
python -m pip install --upgrade pip setuptools wheel

# Install Jupyter & kernel tools inside venv
python -m pip install --upgrade ipykernel jupyterlab

# Install core runtime dependencies (package also declares most deps)
# Install numpy pinned to <2 to avoid PyTorch binary incompatibilities with NumPy 2.x.
python -m pip install "numpy<2" "torch>=2.1" transformers peft safetensors soundfile
if [ $SKIP_AUDIO_PACKAGES -eq 0 ]; then
  python -m pip install librosa torchaudio
fi

# Install the local package in editable mode so notebooks import it directly
python -m pip install -e "$REPO_ROOT/packages/w2vbert_speaker"

# Register an IPython kernel for this venv (user-level)
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo ""
echo "Setup complete. Activate the venv with:"
echo "  source $VENV_PATH/bin/activate"
echo "Or select the kernel named '$KERNEL_DISPLAY_NAME' in JupyterLab / Notebook."

# Print quick checks
python -c "import sys; print('python exe:', sys.executable)"
python -c "import w2vbert_speaker; print('w2vbert_speaker OK, exports:', [n for n in dir(w2vbert_speaker) if not n.startswith('_')])"

exit 0
