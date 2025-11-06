import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
from deeplab.utils.fileio import read_hyperyaml


def _append_to_sys_path(candidate: Path) -> None:
    """Append a directory to sys.path if it exists and is not already present."""
    if candidate.exists():
        resolved = str(candidate)
        if resolved not in sys.path:
            sys.path.append(resolved)


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = next(
    (parent for parent in _THIS_DIR.parents if (parent / "deeplab").exists()),
    _THIS_DIR.parents[3],
)

_append_to_sys_path(_THIS_DIR.parent)  # recipes/DeepASV
_append_to_sys_path(_PROJECT_ROOT)  # repository root
_append_to_sys_path(
    _PROJECT_ROOT / "deeplab/pretrained/audio2vector/module/transformers/src"
)

_possible_toolkit_roots = {
    _PROJECT_ROOT / "audio_toolkit/transforms",
    _PROJECT_ROOT.parent / "audio_toolkit/transforms",
}
for _toolkit_path in _possible_toolkit_roots:
    _append_to_sys_path(_toolkit_path)


_DEFAULT_CONFIG = _PROJECT_ROOT / "recipes/DeepASV/conf/w2v-bert/s3.yaml"
_DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT
    / "deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/model_lmft_0.14.pth"
)


class W2VBERT_SPK_Module(torch.nn.Module):
    def __init__(
        self,
        path: Union[str, Path] = _DEFAULT_CONFIG,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(W2VBERT_SPK_Module, self).__init__()
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = (_PROJECT_ROOT / config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.hparams = read_hyperyaml(str(config_path))
        self.modules = self.hparams["modules"]
        self.model: Optional[torch.nn.Module] = None

    def load_model(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        strict: bool = False,
    ) -> "W2VBERT_SPK_Module":
        checkpoint_path = Path(ckpt_path) if ckpt_path else _DEFAULT_CHECKPOINT
        if not checkpoint_path.is_absolute():
            checkpoint_path = (_PROJECT_ROOT / checkpoint_path).resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        ckpt_data = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        for key, module in self.modules.items():
            if key == "classifier":
                continue
            curr_state_dict = module.state_dict()
            ckpt_state_dict = ckpt_data["modules"].get(key, {})
            mismatched = False
            for name, tensor in curr_state_dict.items():
                if name in ckpt_state_dict and tensor.shape == ckpt_state_dict[name].shape:
                    curr_state_dict[name] = ckpt_state_dict[name]
                else:
                    mismatched = True

            module.load_state_dict(curr_state_dict, strict=strict)
            module = module.to(self.device).eval()

            if mismatched:
                print(f"      {key}: <Partial weights matched>")
            else:
                print(f"      {key}: <All weights matched>")

        self.model = self.modules["spk_model"].to(self.device).eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(
                "Model weights are not loaded. Call `load_model` before inference."
            )

        if x.device != self.device:
            x = x.to(self.device)

        with torch.inference_mode():
            embeddings = self.model(x)

        return embeddings.float().detach()