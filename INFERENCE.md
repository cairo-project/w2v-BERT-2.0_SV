# Inference Guide — W2V-BERT Speaker Encoder

This document explains how to run inference with the W2V-BERT speaker encoder in three ways:

- Eager (Python) using the installed package
- Eager (Python) using a local source checkout
- Scripted (TorchScript) — recommended for distribution or use in environments without the full Python source

It also describes the demo script and the notebook workflow (including kernel setup).

## Quick notes

- A helper package is available at `packages/w2vbert_speaker` and provides runtime helpers:
  - `load_feature_extractor(path_or_dir)` — loads the saved Hugging Face feature extractor
  - `compute_input_features_from_wave(feature_extractor, wave, sr)` — compute HF-style input features from raw waveform and sampling rate (returns a torch.Tensor)
  - `W2VBERT_SPK_Module` — the model wrapper exported by the package (see package API for exact constructor/load methods)
- Preprocessed scripted artifacts (recommended) live as `w2vbert_speaker_script_preprocessed.pt` and expect HF input features (torch.Tensor). The exporter will also save a `feature_extractor/` directory next to the artifact.
- If you plan to use the notebooks, run `./scripts/setup_notebook_env.sh /path/to/python3.11` to create a reproducible venv and register a Jupyter kernel.

## 1) Eager inference using the installed package (recommended for development)

Install the package (editable during development) and required deps inside a venv:

```bash
# from repo root
python -m venv .venv_w2vbert_notebook
source .venv_w2vbert_notebook/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" torch transformers safetensors soundfile
python -m pip install -e packages/w2vbert_speaker
```

Example usage (python):

```python
import librosa
import torch
from w2vbert_speaker import load_feature_extractor, compute_input_features_from_wave, W2VBERT_SPK_Module

# 1) Load a pre-saved HF feature extractor (created by the exporter)
fe = load_feature_extractor('path/to/artifacts/feature_extractor')

# 2) Load waveform and compute input features
wave, sr = librosa.load('audio.wav', sr=16000)
input_features = compute_input_features_from_wave(fe, wave, sr)  # -> torch.Tensor [seq_len, feat_dim] or [1, seq_len, feat_dim]

# 3) Load an eager (Python) model from a checkpoint or from the package API
# The package exposes a model wrapper — consult the package docs for exact loader.
model = W2VBERT_SPK_Module.from_pretrained('/path/to/checkpoint_dir_or_file')
model.eval()

with torch.no_grad():
    # adjust shape if needed to match model expectations (batch dim)
    feats = input_features.unsqueeze(0) if input_features.ndim == 2 else input_features
    embedding = model(feats)
    print('embedding shape:', embedding.shape)
```

Notes:
- The exact loader name for the eager model may vary; check `packages/w2vbert_speaker` `__init__` or README for the model class API. The example above illustrates the typical pattern: load feature extractor, compute HF features, load model, run in eval mode.

## 2) Eager inference using local source (developing the repo)

If you are working inside the repository and want to call the model directly from source (not installed), you can either install the package in editable mode (recommended) or modify PYTHONPATH. The recommended way is:

```bash
python -m venv .venv_w2vbert_notebook
source .venv_w2vbert_notebook/bin/activate
python -m pip install -e packages/w2vbert_speaker
```

Then import exactly as in section (1). Installing editable avoids needing sys.path hacks in notebooks.

## 3) Scripted model inference (TorchScript) — portable and fast

The exporter produces one or more TorchScript artifacts. The most reliable artifact is the "preprocessed" scripted model which expects a tensor of HF input features (instead of raw waveform). The exporter also saves the HF `feature_extractor` next to the artifact so the same preprocessing can be reproduced.

Example usage:

```python
import torch
import librosa
from transformers import AutoFeatureExtractor
from w2vbert_speaker import compute_input_features_from_wave

# Load the scripted model (preprocessed variant)
scripted = torch.jit.load('path/to/w2vbert_speaker_script_preprocessed.pt', map_location='cpu')
scripted.eval()

# Load feature extractor saved by the exporter
fe = AutoFeatureExtractor.from_pretrained('path/to/artifacts/feature_extractor')

# Load audio and compute HF input features (use the helper or compute via the feature extractor)
wave, sr = librosa.load('audio.wav', sr=16000)
feats = compute_input_features_from_wave(fe, wave, sr)  # returns torch.Tensor

with torch.no_grad():
    # ensure batch dimension
    batch = feats.unsqueeze(0) if feats.ndim == 2 else feats
    emb = scripted(batch)
    print('embedding shape', emb.shape)
```

Notes:
- The preprocessed scripted model expects the same HF input features that the eager model consumes. The exporter saves the extractor and documents the expected shapes (batch, seq_len, feat_dim).
- If you have a scripted waveform variant (experimental), its API will accept waveform tensors. Waveform-scripted models are fragile because some preprocessing steps use NumPy; prefer the preprocessed artifact for portability.

## Demo script and notebook

- Create/register a notebook kernel and venv: run `./scripts/setup_notebook_env.sh /path/to/python3.11` from repo root. This creates `.venv_w2vbert_notebook`, installs `numpy<2`, `torch`, `transformers`, `safetensors`, optionally audio packages, and registers a kernel named "W2V-BERT Notebook (.venv_w2vbert_notebook)".
- To run the exporter using the packaged helper and produce a preprocessed scripted artifact:

```bash
# make sure the editable package is installed in the kernel venv
source .venv_w2vbert_notebook/bin/activate
./scripts/run_export_preprocessed.sh
```

- Open the demo notebook (for example `recipes/DeepASV/notebooks/w2vbert_preprocessed_inference_minimal.ipynb`) and select the registered kernel. Follow cells: load the feature extractor from `artifacts/feature_extractor`, compute features, load `w2vbert_speaker_script_preprocessed.pt` and run inference.

## Using the model in an external project

- Recommended approach for production systems: ship the preprocessed TorchScript model + saved HF feature extractor. The consumer only needs PyTorch (libtorch) and the HF feature-extractor code (or a small helper to compute the same input features) to reproduce embeddings.
- Minimal runtime:
  - Install a small set of Python dependencies in the target environment: `torch` and `transformers` (for the feature extractor) or implement the preprocessing in native code if you wish to avoid Python on the consumer side.
  - Load the extractor and scripted model, compute input features for audio, then pass them to the scripted model to obtain embeddings.

## Troubleshooting

- If you see an error like "RuntimeError: Numpy is not available" when calling `tensor.numpy()` or when exporting/loading models, ensure the venv uses a NumPy version compatible with the PyTorch binary (this repo's setup pins `numpy<2`).
- If you need librosa/numba support on macOS, prefer Python 3.11 or 3.12; Python 3.14 currently has limited wheel availability for numba.

## Summary

For portability and reproducible inference in downstream systems, prefer the preprocessed TorchScript artifact paired with the saved feature extractor. For development and debugging, use the eager package (editable install) and the Python model wrapper.
