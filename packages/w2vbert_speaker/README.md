# w2vbert-speaker

`w2vbert-speaker` packages the W2V-BERT speaker embedding module so it can be installed into
a clean virtual environment and reused outside of the original research repository. The
package contains the adapter architecture and configuration artifacts, while model weights
are provided separately at runtime.

## Installation

Clone the repository that contains this package and build it locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install /path/to/packages/w2vbert_speaker
```

## Usage

```python
from w2vbert_speaker import W2VBERT_SPK_Module

model = W2VBERT_SPK_Module(frozen_encoder=False)
model.load_model("/path/to/model_lmft_0.14.pth")
audio = torch.randn(1, 16000)
embeddings = model(audio)

# embeddings.shape -> (1, 256)
```

The Hugging Face base model weights will be downloaded automatically unless you pass a
`model_path` pointing to a local checkpoint directory. The speaker fine-tuning checkpoint
path is provided via `load_model`. If you store the Facebook W2V-BERT weights locally you can
embed them by providing `model_path="/path/to/facebook/w2v-bert-2.0"` when instantiating the
module.
