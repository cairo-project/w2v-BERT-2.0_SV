from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch


def load_feature_extractor(fe_dir: Union[str, Path]):
    """Load a Hugging Face feature extractor saved with `save_pretrained`.

    Returns the AutoFeatureExtractor instance. Import is local to avoid hard
    dependency at module import time.
    """
    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(str(fe_dir))


def compute_input_features_from_wave(
    feature_extractor,
    wave: Union[np.ndarray, torch.Tensor],
    sampling_rate: int | None = None,
) -> torch.Tensor:
    """Compute the HF `input_features` tensor from a raw waveform.

    - `wave` may be a 1-D numpy array or a 1-D torch tensor (shape [T]).
    - `feature_extractor` is a HF FeatureExtractor object loaded via
      `AutoFeatureExtractor.from_pretrained(...)`.
    - Returns a torch.Tensor suitable for the preprocessed scripted model.
    """
    # Ensure numpy array for the HF extractor
    if torch.is_tensor(wave):
        wave_np = wave.detach().cpu().numpy()
    else:
        wave_np = np.asarray(wave, dtype=np.float32)

    # HF extractor expects shape [T] (1-D) or an array-like; pass sampling_rate
    kwargs = dict(return_tensors="pt", padding=False, truncation=False, return_attention_mask=False)
    if sampling_rate is not None:
        kwargs["sampling_rate"] = int(sampling_rate)

    out = feature_extractor(wave_np, **kwargs)
    input_features = out["input_features"].to(dtype=torch.float32)
    return input_features
