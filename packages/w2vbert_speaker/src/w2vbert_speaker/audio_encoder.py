"""Audio encoder helpers built on top of Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    BitsAndBytesConfig,
    Wav2Vec2BertConfig,
    Wav2Vec2BertModel,
)

from .forward_impl import forward_w2v_bert, forward_whisper


def create_bnb_config(
    load_in_8bit: bool = True,
    bnb_8bit_use_double_quant: bool = True,
    bnb_8bit_quant_type: str = "llm_int8",
    bnb_8bit_compute_dtype: str = "bfloat16",
) -> BitsAndBytesConfig:
    """Build a bitsandbytes configuration compatible with Transformers."""
    return BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        bnb_8bit_use_double_quant=bnb_8bit_use_double_quant,
        bnb_8bit_quant_type=bnb_8bit_quant_type,
        bnb_8bit_compute_dtype=getattr(torch, bnb_8bit_compute_dtype),
    )


def create_lora_config(
    model_type: str,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list[str]] = None,
    lora_dropout: float = 0.1,
    bias: str = "none",
) -> LoraConfig:
    """Build a LoRA configuration tailored to the specified encoder type."""
    if model_type == "whisper":
        resolved_target_modules = target_modules or ["q_proj", "v_proj"]
        task_type = "SEQ_CLS"
    elif model_type == "w2v-bert":
        resolved_target_modules = target_modules or ["linear_q", "linear_v"]
        task_type = "FEATURE_EXTRACTION"
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Expected 'whisper' or 'w2v-bert'."
        )

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=resolved_target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def _resolve_model_source(model_name: str, model_path: Optional[Union[str, Path]]) -> tuple[str, bool]:
    if model_path is None:
        return model_name, False
    resolved = str(Path(model_path).expanduser().resolve())
    return resolved, True


def _resolve_config_path(
    model_config_path: Optional[Union[str, Path]],
    model_source: str,
    is_local: bool,
) -> Optional[Path]:
    if model_config_path is None:
        return None

    config_path = Path(model_config_path)
    if not config_path.is_absolute() and is_local:
        config_path = Path(model_source) / config_path
    return config_path.resolve()


class AudioEncoder(nn.Module):
    """Wrapper around Hugging Face encoders tailored for speaker embeddings."""

    PEFT_INDICATORS = ["lora_", "adapter", "prefix", "prompt"]
    SUPPORTED_MODEL_TYPES = ["whisper", "w2v-bert"]

    def __init__(
        self,
        model_name: str = "facebook/w2v-bert-2.0",
        *,
        frozen_encoder: bool = True,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        peft_config: Optional[LoraConfig] = None,
        model_config_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_type = self._infer_model_type(model_name)
        self.model_config_path = model_config_path

        if not frozen_encoder and bnb_config is not None:
            raise ValueError("Cannot combine full fine-tuning with quantization (bnb_config).")

        model_source, is_local = _resolve_model_source(model_name, model_path)
        config_path = _resolve_config_path(model_config_path, model_source, is_local)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_source,
            local_files_only=is_local,
        )

        self._setup_model(model_source, is_local, bnb_config, config_path)

        if peft_config is not None:
            self.encoder = get_peft_model(self.encoder, peft_config)

        if frozen_encoder:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

    def forward(self, aud_inputs: torch.Tensor):
        if self.model_type == "whisper":
            return forward_whisper(self, aud_inputs)
        if self.model_type == "w2v-bert":
            return forward_w2v_bert(self, aud_inputs)
        raise ValueError(
            f"Unsupported model type '{model_name}'. Supported values: {self.SUPPORTED_MODEL_TYPES}."
        )

    def _setup_model(
        self,
        model_source: str,
        is_local: bool,
        bnb_config: Optional[BitsAndBytesConfig],
        config_path: Optional[Path],
    ) -> None:
        if config_path is not None:
            config = Wav2Vec2BertConfig.from_json_file(str(config_path))
            full_model = Wav2Vec2BertModel.from_pretrained(
                model_source,
                config=config,
                local_files_only=is_local,
            )
        else:
            full_model = AutoModel.from_pretrained(
                model_source,
                local_files_only=is_local,
                quantization_config=bnb_config,
            )

        if self.model_type == "whisper":
            self.encoder = full_model.encoder
            self.d_model = self.encoder.config.d_model
            self.n_hidden_states = self.encoder.config.encoder_layers + 1
        elif self.model_type == "w2v-bert":
            self.encoder = full_model
            self.d_model = self.encoder.config.hidden_size
            self.n_hidden_states = self.encoder.config.num_hidden_layers + 1
            if hasattr(self.encoder, "masked_spec_embed"):
                delattr(self.encoder, "masked_spec_embed")
        else:
            raise ValueError(
                f"Unsupported model type '{model_name}'. Supported values: {self.SUPPORTED_MODEL_TYPES}."
            )

    def _infer_model_type(self, model_name: str) -> str:
        name_lower = model_name.lower()
        if "whisper" in name_lower:
            return "whisper"
        if "w2v-bert" in name_lower:
            return "w2v-bert"
        raise ValueError(
            f"Unsupported model name '{model_name}'. Supported types: {self.SUPPORTED_MODEL_TYPES}."
        )

    def _is_peft_parameter(self, param_name: str) -> bool:
        return any(indicator in param_name for indicator in self.PEFT_INDICATORS)

    def _module_has_peft_parameter(self, module: nn.Module) -> bool:
        return any(self._is_peft_parameter(name) for name, _ in module.named_parameters())

    def freeze_encoder(self) -> None:
        self.frozen_encoder = True
        for name, param in self.encoder.named_parameters():
            if not self._is_peft_parameter(name):
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self.frozen_encoder = False
        for name, param in self.encoder.named_parameters():
            if not self._is_peft_parameter(name):
                param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "frozen_encoder", False):
            for module in self.encoder.modules():
                if self._module_has_peft_parameter(module):
                    module.train(mode)
                else:
                    module.eval()
        else:
            self.encoder.train(mode)
        return self
