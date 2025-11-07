# w2vbert-speaker-scripted

Lightweight runtime for the preprocessed TorchScript W2V-BERT speaker encoder.

This package is intended to be a minimal dependency way to ship the scripted model artifact
and a small runtime wrapper to load it. It purposely keeps imports lazy so that downstream
projects only need `torch` (and optionally `transformers` or `torchaudio` for preprocessing).

Usage example

```python
from w2vbert_speaker_scripted import load_scripted, compute_input_features_from_wave, load_feature_extractor
import torch

# Load scripted model
model = load_scripted('artifacts/w2vbert_speaker_script_preprocessed.pt')

# Load feature extractor saved by exporter
fe = load_feature_extractor('artifacts/feature_extractor')

# Compute input features (requires transformers to be installed) and run inference
wave = ...  # numpy array
sr = 16000
feats = compute_input_features_from_wave(fe, wave, sr)
emb = model(feats.unsqueeze(0))
```

Notes

- If you want the runtime to avoid `transformers`, implement a pure-Torch preprocessing pipeline (torchaudio) that reproduces the feature extractor behavior. This package intentionally leaves that as an opt-in improvement.
- The `artifacts/` folder is expected to contain the `w2vbert_speaker_script_preprocessed.pt` artifact and the `feature_extractor/` folder created by the exporter.
