"""Minimal runtime helpers for scripted W2V-BERT speaker artifacts.

Design goals:
- Keep imports lazy so the package can be installed with only `torch` in minimal environments.
- Prefer `torchaudio` for resampling if available; otherwise optionally use HF `transformers` feature extractor.
"""
from typing import Union


def load_scripted(path: str, device: str = "cpu"):
    """Load a TorchScript artifact from `path` and return an eval ScriptModule.

    Args:
        path: path to the .pt scripted file
        device: map_location (e.g., 'cpu' or 'cuda')

    Returns:
        torch.jit.ScriptModule
    """
    import torch

    m = torch.jit.load(path, map_location=device)
    m.eval()
    return m


def load_feature_extractor(fe_dir: str):
    """Load a Hugging Face feature extractor from `fe_dir`.

    If `transformers` is not installed, this raises an informative ImportError.
    """
    try:
        from transformers import AutoFeatureExtractor
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required to load the saved feature extractor. "
            "Install transformers or compute features externally and pass them to the scripted model."
        ) from e

    return AutoFeatureExtractor.from_pretrained(fe_dir)


def compute_input_features_from_wave(feature_extractor_or_dir: Union[str, object], wave, sr: int):
    """Compute HF-style input features from raw waveform.

    This function attempts to use `transformers`'s feature extractor if a path or extractor
    object is provided. It will try to resample using `torchaudio` if available.

    Args:
        feature_extractor_or_dir: either a HF feature-extractor object or the path to a saved extractor dir
        wave: 1-D numpy array or python list or torch tensor containing the waveform
        sr: sampling rate of the provided waveform

    Returns:
        torch.Tensor suitable as input to the preprocessed scripted model.

    Raises:
        ImportError: if neither torchaudio nor transformers are available to compute features.
    """
    # Lazy imports to keep package light
    np = None
    try:
        import numpy as np
    except Exception:
        np = None

    # If the user passed a path, load the HF extractor
    extractor = None
    if isinstance(feature_extractor_or_dir, str):
        extractor = load_feature_extractor(feature_extractor_or_dir)
    else:
        extractor = feature_extractor_or_dir

    # Prefer using transformers to compute input_features since it matches exporter behavior.
    try:
        # Local import so we only require transformers when used
        from transformers import is_torch_available

        if is_torch_available():
            # Use the HF feature extractor (works with numpy arrays)
            import torch

            # Ensure waveform is numpy for HF extractor compatibility
            if isinstance(wave, (list, tuple)):
                wave_np = np.asarray(wave)
            elif hasattr(wave, "numpy"):
                wave_np = wave.numpy()
            else:
                wave_np = np.asarray(wave)

            # Transformers feature extractor will do resampling if needed
            inputs = extractor(wave_np, sampling_rate=sr, return_tensors="pt")
            # Some extractors produce dicts like {'input_features': tensor}
            if "input_features" in inputs:
                return inputs["input_features"].squeeze(0)
            # fallback: try 'input_values'
            if "input_values" in inputs:
                return inputs["input_values"].squeeze(0)

    except Exception:
        # If transformers not installed or extractor failed, continue to torchaudio path
        pass

    # Torchaudio-only path is not implemented fully here because reproducing HF preprocessing
    # exactly requires reproducing the feature extractor logic (filterbanks, normalization, etc.).
    # We therefore raise a helpful error directing users to install transformers or compute
    # features ahead-of-time using the saved extractor.
    raise ImportError(
        "Unable to compute input features from waveform: install 'transformers' or precompute input features "
        "using the saved feature_extractor and pass them to the scripted model. "
        "Alternatively, implement a pure-torch preprocessing pipeline (torchaudio) compatible with the extractor."
    )
