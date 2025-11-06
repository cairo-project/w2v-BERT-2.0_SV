"""Forward implementations for Whisper and W2V-BERT models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput


def _get_model_device_dtype(encoder: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    """Return the device and dtype for a given encoder module."""
    params = next(encoder.parameters())
    return params.device, params.dtype


def forward_whisper(self, aud_inputs: torch.Tensor) -> BaseModelOutput:
    """Forward pass for Whisper encoders."""
    features = self.feature_extractor(
        aud_inputs.cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    device, dtype = _get_model_device_dtype(self.encoder)
    input_features = features.input_features.to(device=device, dtype=dtype)

    x = F.gelu(self.encoder.conv1(input_features))
    x = F.gelu(self.encoder.conv2(x))
    x = x.permute(0, 2, 1)

    seq_len = x.size(1)
    if seq_len > self.encoder.embed_positions.num_embeddings:
        raise ValueError("Input sequence is longer than the positional embedding table")
    pos_embed = self.encoder.embed_positions.weight[:seq_len]
    x = x + pos_embed.unsqueeze(0)

    hidden_states = [x]
    for layer in self.encoder.layers:
        x = layer(x, attention_mask=None, layer_head_mask=None, output_attentions=None)[0]
        hidden_states.append(x)

    x = self.encoder.layer_norm(x)
    hidden_states[-1] = x

    return BaseModelOutput(last_hidden_state=x, hidden_states=tuple(hidden_states))


def forward_w2v_bert(self, aud_inputs: torch.Tensor) -> BaseModelOutput:
    """Forward pass for W2V-BERT encoders."""
    features = self.feature_extractor(
        aud_inputs.cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    device, dtype = _get_model_device_dtype(self.encoder)
    input_features = features.input_features.to(device=device, dtype=dtype)

    x = self.encoder.feature_projection(input_features)[0]

    hidden_states = [x]
    for layer in self.encoder.encoder.layers:
        x = layer(x)[0]
        hidden_states.append(x)

    return BaseModelOutput(last_hidden_state=x, hidden_states=tuple(hidden_states))
