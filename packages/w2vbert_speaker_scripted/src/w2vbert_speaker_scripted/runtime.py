"""Minimal runtime helpers for scripted W2V-BERT speaker artifacts.

Design goals:
- Keep imports lazy so the package can be installed with only `torch` in minimal environments.
- Prefer `torchaudio` for resampling if available; otherwise optionally use HF `transformers` feature extractor.
"""
from typing import Union, Optional
from pathlib import Path

import torch
import numpy as np
from transformers import AutoFeatureExtractor


def load_scripted(path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    """Load a TorchScript artifact from `path` and return an eval ScriptModule.

    Args:
        path: path to the .pt scripted file
        device: map_location (e.g., 'cpu' or 'cuda')

    Returns:
        torch.jit.ScriptModule
    """
    # Try loading directly onto device (best for CUDA). If that fails, fall back to CPU load and .to(device).
    try:
        m = torch.jit.load(str(path), map_location=device)
        m.eval()
        return m
    except Exception:
        m = torch.jit.load(str(path), map_location="cpu")
        m.eval()
        try:
            m = m.to(device)
        except Exception:
            # ignore move errors; caller can still use CPU
            pass
        return m


def load_feature_extractor(fe_dir: str) -> AutoFeatureExtractor:
    """Load a Hugging Face feature extractor from `fe_dir`.

    transformers is a hard dependency for this runtime; call will raise if not present.
    """
    return AutoFeatureExtractor.from_pretrained(str(fe_dir))


def compute_input_features_from_wave(feature_extractor_or_dir: Union[str, object], wave, sr: Optional[int]):
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
    # Always use Hugging Face extractor (hard dependency) to guarantee deterministic parity.
    if isinstance(feature_extractor_or_dir, str):
        extractor = load_feature_extractor(feature_extractor_or_dir)
    else:
        extractor = feature_extractor_or_dir

    # Ensure numpy input for the HF extractor
    if isinstance(wave, (list, tuple)):
        wave_np = np.asarray(wave)
    elif hasattr(wave, "numpy"):
        wave_np = wave.numpy()
    else:
        wave_np = np.asarray(wave)

    # If sampling rate not provided, fall back to extractor sampling_rate
    sampling_rate = sr if sr is not None else getattr(extractor, "sampling_rate", None)

    inputs = extractor(wave_np, sampling_rate=sampling_rate, return_tensors="pt")
    if "input_features" in inputs:
        return inputs["input_features"].squeeze(0)
    if "input_values" in inputs:
        return inputs["input_values"].squeeze(0)
    raise RuntimeError("Feature extractor did not produce 'input_features' or 'input_values'")


class W2VBERT_SPK_Scripted(torch.nn.Module):
    """Plug-and-play replacement for W2VBERT_SPK_Module that wraps a scripted artifact.

    Constructor args:
      - scripted_path: path to the TorchScript .pt artifact
      - feature_extractor_dir: path to saved Hugging Face feature_extractor (required)
      - device: optional device string or torch.device (defaults to CUDA if available else CPU)

    The class implements forward(self, x: torch.Tensor) -> torch.Tensor. It accepts either
    raw waveform (1D or [B, T]) or precomputed HF input_features (shape [seq_len, feat_dim] or [B, seq_len, feat_dim]).
    """

    def __init__(
        self,
        *,
        scripted_path: str,
        feature_extractor_dir: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.feature_extractor = load_feature_extractor(feature_extractor_dir)
        # feature size heuristic
        self.feature_size = int(getattr(self.feature_extractor, "feature_size", getattr(self.feature_extractor, "num_mel_bins", 80)))

        # load scripted model
        self.scripted = load_scripted(scripted_path, device=str(self.device))
        try:
            self.scripted.to(self.device)
        except Exception:
            # some scripted modules may not support .to for all devices
            pass

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        try:
            self.scripted.to(self.device)
        except Exception:
            pass
        return self

    def compute_input_features(self, wave, sr: Optional[int] = None) -> torch.Tensor:
        inp = compute_input_features_from_wave(self.feature_extractor, wave, sr)
        return inp.to(dtype=torch.float32, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept waveform or precomputed features and return embeddings."""
        # If input is not a tensor, convert
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        # Heuristic: if last dim matches feature_size and rank >=2, assume precomputed HF features
        is_features = x.dim() >= 2 and x.shape[-1] == self.feature_size

        if is_features:
            # Normalize features to [B, L, D]
            input_features = x
            if input_features.dim() == 1:
                input_features = input_features.unsqueeze(0).unsqueeze(0)
            elif input_features.dim() == 2:
                # [L, D] -> [1, L, D]
                input_features = input_features.unsqueeze(0)
            input_features = input_features.to(dtype=torch.float32, device=self.device)
        else:
            # Treat as waveform input. compute_input_features accepts 1D/2D arrays or tensors and
            # returns [L, D] (squeezed batch dim). Move result to desired device inside that call.
            input_features = self.compute_input_features(x, sr=getattr(self.feature_extractor, "sampling_rate", None))
            # Normalize to [B, L, D]
            if input_features.dim() == 1:
                input_features = input_features.unsqueeze(0).unsqueeze(0)
            elif input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)

        # Final safety: ensure a 3D tensor [B, L, D] is passed to the scripted module.
        if input_features.dim() < 3:
            # This is defensive: reshape whatever we have into [1, L, D]
            input_features = input_features.view(1, *input_features.shape)

        # Ensure dtype and device
        input_features = input_features.to(dtype=torch.float32)
        if input_features.device != self.device:
            try:
                input_features = input_features.to(self.device)
            except Exception:
                # If move fails, keep as-is (caller may still use CPU)
                pass

        # Prevent accidental huge-batch / huge-seq misinterpretation: if the first dim is much larger
        # than the sequence dim, it is likely the caller passed unbatched features; we can't automatically
        # fix semantics, but warn to help debugging.
        try:
            bsz, seqlen, _ = input_features.shape
            if bsz > seqlen * 4:
                print(f"Warning: input_features batch size ({bsz}) >> sequence length ({seqlen});\n" \
                      "this may indicate a mis-shaped input (seq_len treated as batch).")
        except Exception:
            # ignore shape inspection errors
            pass

        with torch.inference_mode():
            out = self.scripted(input_features)

        # scripted may return tensor or tuple; return first tensor as embedding
        emb = out[0] if isinstance(out, (list, tuple)) else out
        return emb.float().detach()

