"""Speaker embedding backbones used by the inference module."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..audio_encoder import AudioEncoder
from .modules import pooling_v2


class Audio2VecBasedAdapter(nn.Module):
    """Audio2Vec backbone with adapter layers and attentive pooling."""

    def __init__(
        self,
        *,
    model_name: str = "facebook/w2v-bert-2.0",
    model_path: Optional[str] = None,
        frozen_encoder: bool = False,
        bnb_config=None,
        peft_config=None,
    encoder_config_path: Optional[str] = None,
        n_mfa_layers: int = -1,
        pooling_layer: str = "ASP",
        embd_dim: int = 256,
        adapter_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.front = AudioEncoder(
            model_name=model_name,
            frozen_encoder=frozen_encoder,
            bnb_config=bnb_config,
            peft_config=peft_config,
            model_config_path=encoder_config_path,
            model_path=model_path,
        )
        self.drop = nn.Dropout(dropout) if dropout else None

        if n_mfa_layers == -1:
            self.n_mfa_layers = self.front.n_hidden_states
        else:
            self.n_mfa_layers = n_mfa_layers
        if not (1 <= self.n_mfa_layers <= self.front.n_hidden_states):
            raise ValueError("n_mfa_layers must be between 1 and the encoder depth")

        self.adapter_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.front.d_model, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.ReLU(True),
                    nn.Linear(adapter_dim, adapter_dim),
                )
                for _ in range(self.n_mfa_layers)
            ]
        )

        feat_dim = adapter_dim * self.n_mfa_layers
        pooling_cls = getattr(pooling_v2, pooling_layer.upper(), None)
        if pooling_cls is None:
            raise ValueError(f"Unknown pooling layer '{pooling_layer}'.")
        self.pooling = pooling_cls(feat_dim, adapter_dim)
        self.bottleneck = nn.Linear(feat_dim * self.pooling.expansion, embd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.front(x)
        if self.n_mfa_layers == 1:
            hidden = features.last_hidden_state
        else:
            hidden_states = features.hidden_states[-self.n_mfa_layers :]
            projected = [layer(h) for layer, h in zip(self.adapter_layers, hidden_states)]
            hidden = torch.cat(projected, dim=-1)

        pooled = self.pooling(hidden)
        if self.drop is not None:
            pooled = self.drop(pooled)
        return self.bottleneck(pooled)
