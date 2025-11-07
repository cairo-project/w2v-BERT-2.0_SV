#!/usr/bin/env python3
"""Export the packaged W2V-BERT speaker encoder to TorchScript."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "recipes").exists() and (candidate / "deeplab").exists():
            return candidate
    raise RuntimeError("Unable to locate the repository root. Run the script from inside the repo.")


def resolve_checkpoint(repo_root: Path, override: Path | None) -> Path:
    if override is not None:
        resolved = override if override.is_absolute() else (repo_root / override).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved}")
        return resolved

    from recipes.DeepASV.utils.inference import get_checkpoint_candidates

    for candidate in get_checkpoint_candidates():
        if candidate.exists():
            return candidate

    expected = "\n".join(f"  - {candidate}" for candidate in get_checkpoint_candidates())
    raise FileNotFoundError(
        "No checkpoint found. Provide --checkpoint or place weights at one of these paths:\n" + expected
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for the finetuned adapter weights (defaults to the canonical locations).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="TorchScript artifact path. Defaults to packages/w2vbert_speaker/artifacts/w2vbert_speaker_script.pt.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to W2VBERT_SPK_Module during export (default: cpu).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Directory or file containing the base encoder weights (model.safetensors). Defaults to local cache if present.",
    )
    parser.add_argument(
        "--example-seconds",
        type=float,
        default=5.0,
        help="Length of the synthetic waveform (in seconds) used for tracing.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Precompute feature-extractor outputs and export a scripted module that accepts precomputed features (avoids tracing numpy conversion).",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading checkpoint weights.",
    )
    parser.add_argument(
        "--install-package",
        action="store_true",
        help="If the package is not importable, attempt `pip install -e packages/w2vbert_speaker` into the current env.",
    )
    return parser.parse_args()


def determine_sample_rate(module) -> int:
    spk_module = module.modules_dict.get("spk_model")
    if spk_module is None:
        return 16000
    front = getattr(spk_module, "front", None)
    extractor = getattr(front, "feature_extractor", None)
    sr = getattr(extractor, "sampling_rate", None)
    if sr is None:
        return 16000
    return int(sr)


def _model_search_candidates(repo_root: Path) -> Iterable[Path]:
    target = Path("audio2vector/ckpts/facebook/w2v-bert-2.0")
    yield repo_root / "deeplab/pretrained" / target
    yield repo_root / "pretrained" / target
    yield repo_root.parent / "pretrained" / target


def resolve_model_path(repo_root: Path, override: Path | None) -> Tuple[Path, Path]:
    if override is not None:
        candidate = override if override.is_absolute() else (repo_root / override).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Model path not found: {candidate}")
        resolved_dir = candidate if candidate.is_dir() else candidate.parent
        weights_file = candidate if candidate.is_file() else resolved_dir / "model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"model.safetensors not found at {weights_file}")
        return resolved_dir, weights_file

    for root_candidate in _model_search_candidates(repo_root):
        weights_file = (root_candidate / "model.safetensors").resolve()
        if weights_file.exists():
            return root_candidate.resolve(), weights_file

    expected = "\n".join(f"  - {(candidate / 'model.safetensors').resolve()}" for candidate in _model_search_candidates(repo_root))
    raise FileNotFoundError(
        "No local base model found. Provide --model-path or place model.safetensors at one of these locations:\n" + expected
    )


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)

    # Prefer the package to be installed in the active environment. Try importing
    # the public module first; if it fails and --install-package is set, install
    # the local package in editable mode and retry. Only fall back to modifying
    # sys.path when neither approach succeeds.
    try:
        importlib.import_module("w2vbert_speaker.module")
    except Exception:
        if args.install_package:
            pkg_path = (repo_root / "packages" / "w2vbert_speaker").resolve()
            print(f"Package not importable; attempting to pip install -e {pkg_path}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(pkg_path)])
            importlib.invalidate_caches()
            try:
                importlib.import_module("w2vbert_speaker.module")
            except Exception as exc:  # pragma: no cover - environment install failed
                raise RuntimeError(f"Failed to import w2vbert_speaker after installation: {exc}")
        else:
            # Do not modify sys.path as a runtime fallback. Require the package to be
            # importable or ask the user to run the installer. Fail fast to avoid
            # hidden behavior in CI or production environments.
            raise RuntimeError(
                "Package 'w2vbert_speaker' is not importable. "
                "Install it (e.g. `pip install -e packages/w2vbert_speaker`) or rerun with --install-package to attempt installation."
            )

    from w2vbert_speaker.module import W2VBERT_SPK_Module

    checkpoint = resolve_checkpoint(repo_root, args.checkpoint)
    model_dir, weights_file = resolve_model_path(repo_root, args.model_path)

    output_path = (
        args.output
        if args.output is not None
        else repo_root / "packages/w2vbert_speaker/artifacts/w2vbert_speaker_script.pt"
    )
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer a local config.json next to the model weights if present. If the
    # base model's config is missing model-specific pruning fields expected by
    # the local Transformers fork, merge missing fields from the packaged
    # pruning config into a temporary config file and pass that to the module.
    local_config = (model_dir / "config.json").resolve()
    encoder_config_to_use: str | None = None
    if local_config.exists():
        try:
            # load base config
            with open(local_config, "r", encoding="utf-8") as fh:
                base_cfg = json.load(fh)
        except Exception:
            base_cfg = {}

        # load packaged prune config for missing keys
        try:
            from importlib import resources

            packaged_cfg_path = resources.files("w2vbert_speaker.data").joinpath("config_prune_tea.json")
            with open(packaged_cfg_path, "r", encoding="utf-8") as fh:
                packaged_cfg = json.load(fh)
        except Exception:
            packaged_cfg = {}

        # If packaged config has keys missing from base, merge them (non-destructively)
        merged = dict(base_cfg)
        for key, val in packaged_cfg.items():
            if key not in merged:
                merged[key] = val

        # If merged differs from base, write a temp config next to the artifact and use it.
        if merged != base_cfg:
            # Normalize per-layer lists so their lengths match num_hidden_layers
            try:
                nh = int(merged.get("num_hidden_layers") or base_cfg.get("num_hidden_layers") or 0)
            except Exception:
                nh = 0
            if nh > 0:
                for k, v in list(merged.items()):
                    if isinstance(v, list) and len(v) > 0 and len(v) < nh:
                        # Only extend lists whose elements are primitives or lists (likely per-layer configs)
                        if all(not isinstance(x, dict) for x in v):
                            last = v[-1]
                            merged[k] = v + [last] * (nh - len(v))

            temp_cfg_path = output_path.parent / "merged_encoder_config.json"
            with open(temp_cfg_path, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, indent=2)
            encoder_config_to_use = str(temp_cfg_path)
            print(f"Using encoder assets from: {model_dir} (merged config written to: {temp_cfg_path})")
        else:
            encoder_config_to_use = str(local_config)
            print(f"Using encoder assets from: {model_dir} (config: {local_config})")
    else:
        print(f"Using encoder assets from: {model_dir} (no local config.json; using packaged encoder config)")

    if encoder_config_to_use is not None:
        module = W2VBERT_SPK_Module(device=args.device, model_path=str(model_dir), encoder_config_path=encoder_config_to_use)
    else:
        module = W2VBERT_SPK_Module(device=args.device, model_path=str(model_dir))
    module.load_model(ckpt_path=checkpoint, strict=args.strict_load)
    module.eval()

    # If present, save the feature-extractor next to the artifact so runtimes can
    # load it without constructing the full eager module. This stores only the
    # preprocessing config (no model weights).
    try:
        spk_front = module.modules_dict.get("spk_model").front
    except Exception:
        spk_front = None
    feature_extractor_dir = None
    if spk_front is not None:
        fe = getattr(spk_front, "feature_extractor", None)
        if fe is not None:
            feature_extractor_dir = output_path.parent / "feature_extractor"
            feature_extractor_dir.mkdir(parents=True, exist_ok=True)
            try:
                fe.save_pretrained(str(feature_extractor_dir))
                print(f"Saved feature_extractor to: {feature_extractor_dir}")
            except Exception as exc:
                print(f"Warning: failed to save feature_extractor: {exc}")

    sample_rate = determine_sample_rate(module)
    num_samples = max(int(sample_rate * args.example_seconds), sample_rate)
    example = torch.zeros(1, num_samples, dtype=torch.float32, device=module.device)

    if args.preprocess:
        # Precompute feature extractor outputs for the example waveform and
        # export a wrapper that accepts precomputed input_features tensors.
        front = module.modules_dict["spk_model"].front
        # feature_extractor expects numpy inputs; run it once now (outside trace).
        # If NumPy is not available in this environment, give a clear error
        # with instructions to install it. Converting torch tensors to numpy
        # requires NumPy.
        try:
            import numpy as _np  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment missing numpy
            raise RuntimeError(
                "Preprocessing requires NumPy but it is not available in this Python environment. "
                f"Install it and retry, e.g. `{sys.executable} -m pip install numpy`, or run the exporter without --preprocess. "
                "(The --preprocess path runs the Hugging Face feature extractor which currently expects NumPy inputs.)"
            ) from exc

        # safe to convert to numpy now
        example_np = example.cpu().numpy()
        features = front.feature_extractor(example_np, sampling_rate=sample_rate, return_tensors="pt", padding=False, truncation=False, return_attention_mask=False)
        input_features = features["input_features"].to(dtype=next(module.parameters()).dtype)

        class FeatureWrapper(torch.nn.Module):
            def __init__(self, spk_model):
                super().__init__()
                self.encoder = spk_model.front.encoder
                self.n_mfa_layers = spk_model.n_mfa_layers
                self.adapter_layers = spk_model.adapter_layers
                self.pooling = spk_model.pooling
                self.bottleneck = spk_model.bottleneck
                self.drop = spk_model.drop

            def forward(self, input_features_tensor: torch.Tensor) -> torch.Tensor:
                # input_features_tensor: [batch, seq_len, feat_dim]
                x = self.encoder.feature_projection(input_features_tensor)[0]
                hidden_states = [x]
                for layer in self.encoder.encoder.layers:
                    x = layer(x)[0]
                    hidden_states.append(x)

                if self.n_mfa_layers == 1:
                    hidden = hidden_states[-1]
                else:
                    hidden_states_slice = hidden_states[-self.n_mfa_layers :]
                    projected = [layer(h) for layer, h in zip(self.adapter_layers, hidden_states_slice)]
                    hidden = torch.cat(projected, dim=-1)

                pooled = self.pooling(hidden)
                if self.drop is not None:
                    pooled = self.drop(pooled)
                return self.bottleneck(pooled)

        wrapper = FeatureWrapper(module.modules_dict["spk_model"]).eval()
        # Probe reference embedding by running wrapper on example input_features
        with torch.inference_mode():
            reference = wrapper(input_features)

        try:
            scripted = torch.jit.script(wrapper)
            print("Exported feature-wrapper using torch.jit.script")
        except Exception:
            scripted = torch.jit.trace(wrapper, input_features, strict=False)
            print("Exported feature-wrapper using torch.jit.trace")

        scripted = scripted.cpu()
        # adjust output artifact name to indicate preprocessed interface
        output_path = output_path.with_name(output_path.stem + "_preprocessed" + output_path.suffix)
        preprocessed_flag = True
    else:
        with torch.inference_mode():
            reference = module(example)

        # Try to use torch.jit.script (preferred) which preserves control flow and
        # avoids the tracer issues around converting tensors to NumPy. If scripting
        # fails (many third-party libs are not scriptable), fall back to tracing.
        try:
            scripted = torch.jit.script(module)
            print("Exported model using torch.jit.script")
        except Exception as exc:  # pragma: no cover - fallback
            print(f"torch.jit.script failed: {exc}; falling back to torch.jit.trace")
            scripted = torch.jit.trace(module, example, strict=False)
        scripted = scripted.cpu()

    metadata: Dict[str, str] = {
        "sample_rate": str(sample_rate),
        "embedding_dim": str(int(reference.shape[-1])),
        "checkpoint_path": str(checkpoint),
        "model_path": str(weights_file),
    }
    if 'preprocessed_flag' in locals() and preprocessed_flag:
        metadata["preprocessed"] = "true"
    extra_files = {key: value.encode("utf-8") for key, value in metadata.items()}

    # Use the internal name expected by this PyTorch build for extra files
    torch.jit.save(scripted, str(output_path), _extra_files=extra_files)

    print(f"TorchScript module saved to: {output_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
