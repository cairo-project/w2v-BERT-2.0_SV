# ARTIFACTS — w2v-bert-2.0 Speaker Embedding

This document lists the prebuilt artifacts the notebooks and packaged runtimes expect, where to download them, and short examples showing how to point the eager and scripted packages at those artifacts.

Download

The exported artifacts can be downloaded from the project drive folder:

https://drive.google.com/drive/u/0/folders/1bz_GmREFNNTuIPrsAEJsyAp7CFwQ_vIz

Required layout (repo-relative)

The notebooks in `recipes/DeepASV/notebooks` assume the following directories exist relative to the repository root (`REPO_ROOT`):

- Eager (packaged) artifacts:
  - `../pretrained/w2vbert_speaker_eager`
  - Expected contents: `model_lmft_0.14.pth`, any packaged eager files needed by `w2vbert_speaker`.

- Scripted artifacts (TorchScript + HF feature-extractor):
  - `../pretrained/w2vbert_speaker_scripted`
  - Expected contents:
    - `w2vbert_speaker_script_preprocessed.pt` (preprocessed TorchScript artifact)
    - `w2vbert_speaker_script.pt` (optional waveform-entry scripted artifact)
    - `feature_extractor/` (Hugging Face feature_extractor saved dir used by the scripted runtime)

- Dataset (example used by notebooks):
  - `../datasets/voxceleb1test`
  - Typical audio path used in demos: `../datasets/voxceleb1test/wav/id10270/5r0dWxy17C8/00001.wav`

Notes

- The notebooks were updated to use the fixed repo-relative paths above so they do not "guess" locations.
- If you place artifacts in other locations, update the `EAGER_DIR` / `SCRIPTED_DIR` / `DATASET_DIR` variables inside the notebook or provide equivalent environment / wrapper logic before calling.

Examples: specifying artifact paths when running inference

1) Eager packaged model (python snippet)

```python
from pathlib import Path
from recipes.DeepASV.utils.inference import W2VBERT_SPK_Module
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust to your script location
EAGER_DIR = (REPO_ROOT.parent / "pretrained" / "w2vbert_speaker_eager").resolve()
checkpoint = EAGER_DIR / "model_lmft_0.14.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = W2VBERT_SPK_Module(device=device, model_path=str(EAGER_DIR)).load_model(checkpoint)
```

2) Scripted packaged runtime (python snippet)

```python
from pathlib import Path
from w2vbert_speaker_scripted import W2VBERT_SPK_Scripted
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTED_DIR = (REPO_ROOT.parent / "pretrained" / "w2vbert_speaker_scripted").resolve()
scripted_artifact = SCRIPTED_DIR / "w2vbert_speaker_script_preprocessed.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = W2VBERT_SPK_Scripted(scripted_path=str(scripted_artifact), device=device)
# wrapper accepts a waveform tensor shaped [1, T]
```

Troubleshooting

- If a notebook errors with "file not found", ensure the `pretrained` and `datasets` directories are present at the repo parent level and contain the expected artifacts.
- For parity checks, use the preprocessed scripted artifact (`w2vbert_speaker_script_preprocessed.pt`) — it is produced to give near-exact numeric parity to the eager model.

License & distribution

The artifacts hosted in the Drive folder were exported for use with this repository. Check their export/readme inside the Drive folder for licensing details.
