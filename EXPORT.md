# Export Guide — TorchScript for W2V-BERT Speaker Encoder

This document explains how the repository exports W2V-BERT speaker encoder models to TorchScript, why the export is implemented the way it is, and what trade-offs and implications to expect.

Files and scripts used by the exporter

- `scripts/export_w2vbert_torchscript.py` — main exporter script. Supports exporting a waveform-accepting TorchScript artifact and a more robust `preprocessed` artifact (which accepts HF input-features instead of raw waveform).
- `scripts/run_export_preprocessed.sh` — convenience wrapper that creates an environment (or uses an existing one), installs the package, and runs the exporter in preprocessed mode.
- `packages/w2vbert_speaker/artifacts/` — exporter writes artifacts here (scripted model files and `feature_extractor/`).

Why we export and the two supported modes

1. Preprocessed (recommended):
   - The exporter saves a TorchScript artifact that accepts HF-style input features (a torch.Tensor of shape [batch, seq_len, feat_dim]).
   - The exporter also saves the Hugging Face FeatureExtractor (via `fe.save_pretrained(...)`) next to the artifact so consumers can compute identical input features.
   - Pros: deterministic parity with eager model, robust (avoids NumPy <-> torch conversions in traced code), portable, smaller API surface.
   - Cons: requires a small preprocessing step prior to inference (compute input features). This step is straightforward and can be implemented with the saved HF extractor.

2. Waveform (experimental):
   - The exporter attempts to produce a scripted model that accepts raw waveform tensors and performs preprocessing internally.
   - In practice the repository's preprocessing pipeline uses Hugging Face feature extractors which, under the hood, often convert tensors to NumPy arrays. These conversions are not fully scriptable and cause tracer/scripting to bake constants or fail to capture dynamic paths.
   - Pros: single artifact that takes raw waveform inputs.
   - Cons: fragile — tracing may capture numpy conversion behavior, produce mismatched outputs vs eager, or fail for certain transformers/feature-extractor implementations. Because of this fragility the waveform-scripted artifact is experimental and not recommended for distribution.

How the exporter works (high level)

1. Load the base W2V-BERT encoder weights and any adapter/checkpoint you provide.
2. Merge/normalize model configuration if a pruning/config override is detected (for example when using pruned adapters). This ensures layer lists match `num_hidden_layers`.
3. Save the Hugging Face `feature_extractor` using `fe.save_pretrained(...)` so that preprocessing can be reproduced exactly by consumers.
4. Build a small wrapper (FeatureWrapper) that accepts either raw waveform or precomputed HF features. For the preprocessed export the wrapper accepts HF features and forwards them into the encoder. The wrapper is scripted via `torch.jit.script` when possible; if scripting fails a fallback `torch.jit.trace` with careful example inputs is used.
5. Use `torch.jit.save(..., _extra_files=...)` to persist both the artifact and small metadata files (e.g., serialized config snippets and the exported extractor directory).

Notes about reproducibility and numeric parity

- The preprocessed TorchScript artifact has been validated to produce identical embeddings to the eager model when provided with the same HF input features (cosine=1.0, L2=0.0 in test comparisons). This is because the exported graph avoids Python/NumPy preprocessing steps that can change numeric paths between eager and traced outputs.
- Waveform-scripted outputs often differ from eager outputs due to tracing of NumPy conversions and/or implicit casting; this explains mismatches found during testing.

Runtime & portability implications

- Numpy / PyTorch compatibility: a common runtime failure is "RuntimeError: Numpy is not available" or ABI incompatibilities when the environment has NumPy 2.x but the PyTorch wheel was compiled against NumPy 1.x. To avoid this, install `numpy<2` in the environment that runs the exporter and in the environment that loads scripted artifacts with tensor.numpy() calls.
- Feature extractor versions: the exporter saves the feature extractor used at export time. Consumers must use the same extractor artifacts (or a fully compatible extractor) to guarantee identical input features.
- Dependency surface on consumer side: preprocessed scripted model consumers need only `torch` (to load the scripted artifact) and the HF feature extractor code (or a small helper to perform the same feature extraction). If you want to avoid Python and use libtorch, implement the same preprocessing in your target language or serialize precomputed features.

Best practices

- Prefer exporting the preprocessed TorchScript artifact and shipping it with the saved `feature_extractor/` directory. This enables consumers to run inference deterministically with minimal dependencies.
- Pin numpy to `<2` in the venv used for export to avoid PyTorch/NumPy ABI mismatches.
- When exporting models for other teams, include a small README or metadata file describing the expected input shapes and the extractor commit/version used for preprocessing.

Common pitfalls and troubleshooting

- If scripted outputs differ from eager outputs:
  - Prefer the preprocessed artifact (re-export using `--preprocess` flag).
  - Ensure the saved feature extractor is used to compute input features.
  - Verify model config merges (if using pruned adapters) — inconsistent per-layer lists vs `num_hidden_layers` can cause shape or behavior mismatches.
- If installation of `librosa`/`numba` fails on the target Python (e.g. Python 3.14), use Python 3.11/3.12 or skip audio packages and compute features in a different environment; the exporter script includes logic to skip these packages when only an incompatible Python is available.

Commands (examples)

Export a preprocessed scripted model using the provided wrapper:

```bash
# create a venv (recommended with Python 3.11/3.12), activate it and install deps
./scripts/setup_notebook_env.sh /path/to/python3.11
source .venv_w2vbert_notebook/bin/activate

# run the exporter (script will save artifacts under the package artifacts dir)
python scripts/export_w2vbert_torchscript.py --preprocess --model-path ../pretrained/model.safetensors --output packages/w2vbert_speaker/artifacts

# or use the convenience wrapper
./scripts/run_export_preprocessed.sh
```

Loading the exported preprocessed artifact (consumer):

```python
import torch
from transformers import AutoFeatureExtractor
from w2vbert_speaker import compute_input_features_from_wave

scripted = torch.jit.load('packages/w2vbert_speaker/artifacts/w2vbert_speaker_script_preprocessed.pt')
fe = AutoFeatureExtractor.from_pretrained('packages/w2vbert_speaker/artifacts/feature_extractor')

# compute features and run
wave, sr = ...  # load audio
feats = compute_input_features_from_wave(fe, wave, sr)
emb = scripted(feats.unsqueeze(0))
```

Final thoughts

Exporting deep models to TorchScript for downstream consumption is extremely useful for distribution and production deployment. However, when the preprocessing pipeline uses Python-level libraries that internally convert to NumPy, tracing/scriptability becomes fragile. The preprocessed export pattern used here isolates preprocessing from the scripted graph and yields deterministic parity with the eager model while keeping the runtime lightweight.
