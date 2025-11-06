"""Public-facing module that exposes the W2V-BERT speaker encoder."""

from __future__ import annotations

import warnings
from importlib import resources
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from .local.spk_model import Audio2VecBasedAdapter


def _default_encoder_config() -> Path:
    traversable = resources.files("w2vbert_speaker.data").joinpath("config_prune_tea.json")
    return Path(str(traversable))


class W2VBERT_SPK_Module(nn.Module):
    """Wrapper that builds the speaker embedding stack and loads finetuned weights."""

    def __init__(
        self,
        *,
        model_name: str = "facebook/w2v-bert-2.0",
        model_path: Union[str, Path, None] = None,
        encoder_config_path: Union[str, Path, None] = None,
        device: Optional[Union[str, torch.device]] = None,
        n_mfa_layers: int = -1,
        embedding_dim: int = 256,
        adapter_dim: int = 128,
        dropout: float = 0.0,
    frozen_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        resolved_config = (
            Path(encoder_config_path).expanduser().resolve()
            if encoder_config_path is not None
            else _default_encoder_config()
        )
        if not resolved_config.exists():
            raise FileNotFoundError(f"Encoder config not found at {resolved_config}")

        resolved_model_path = (
            Path(model_path).expanduser().resolve() if model_path is not None else None
        )

        self.modules_dict: dict[str, nn.Module] = {
            "spk_model": Audio2VecBasedAdapter(
                model_name=model_name,
                model_path=str(resolved_model_path) if resolved_model_path else None,
                frozen_encoder=frozen_encoder,
                encoder_config_path=str(resolved_config),
                n_mfa_layers=n_mfa_layers,
                embd_dim=embedding_dim,
                adapter_dim=adapter_dim,
                dropout=dropout,
            )
        }
        self.model: Optional[nn.Module] = None

    def load_model(
        self,
        ckpt_path: Union[str, Path],
        *,
        strict: bool = False,
    ) -> "W2VBERT_SPK_Module":
        checkpoint_path = Path(ckpt_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # If a base model (e.g. model.safetensors + config.json) lives next to the
        # provided checkpoint, prefer loading the encoder from that local folder so
        # transformers does not attempt to download artifacts from the hub.
        candidate_model_dir = checkpoint_path.parent
        try:
            has_local_base = any(
                (candidate_model_dir / fname).exists()
                for fname in ("model.safetensors", "pytorch_model.bin", "config.json")
            )
        except Exception:
            has_local_base = False

        if has_local_base:
            try:
                front = self.modules_dict["spk_model"].front
                # reconfigure the front encoder to use the local model directory
                config_path = getattr(front, "model_config_path", None)
                front._setup_model(str(candidate_model_dir), True, None, Path(config_path) if config_path else None)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Failed to reconfigure local encoder from {candidate_model_dir}: {exc}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        modules_state = checkpoint.get("modules") or {}
        if "spk_model" not in modules_state:
            raise KeyError("Checkpoint does not contain a 'spk_model' entry under 'modules'.")

        module = self.modules_dict["spk_model"]
        current_state = module.state_dict()
        ckpt_state = modules_state["spk_model"]
        mismatched: list[str] = []

        for name, tensor in current_state.items():
            if name in ckpt_state and ckpt_state[name].shape == tensor.shape:
                current_state[name] = ckpt_state[name]
            else:
                mismatched.append(name)

        module.load_state_dict(current_state, strict=strict)
        module = module.to(self.device).eval()
        self.modules_dict["spk_model"] = module
        self.model = module

        if mismatched:
            warnings.warn(
                "Partial weight load detected for spk_model: " + ", ".join(mismatched),
                RuntimeWarning,
            )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model weights are not loaded. Call 'load_model' first.")
        if x.device != self.device:
            x = x.to(self.device)
        with torch.inference_mode():
            embeddings = self.model(x)
        return embeddings.float().detach()
