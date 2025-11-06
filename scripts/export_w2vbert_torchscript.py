#!/usr/bin/env python3
"""Export the packaged W2V-BERT speaker encoder to TorchScript."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "recipes").exists() and (candidate / "deeplab").exists():
            return candidate
    raise RuntimeError("Unable to locate the repository root. Run the script from inside the repo.")


def extend_sys_path(repo_root: Path) -> None:
    additions = [
        repo_root / "packages/w2vbert_speaker/src",
        repo_root / "recipes/DeepASV",
        repo_root / "deeplab/pretrained/audio2vector/module/transformers/src",
        repo_root,
    ]
    for candidate in additions:
        if candidate.exists():
            resolved = str(candidate)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)


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
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading checkpoint weights.",
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
    extend_sys_path(repo_root)

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

    sample_rate = determine_sample_rate(module)
    num_samples = max(int(sample_rate * args.example_seconds), sample_rate)
    example = torch.zeros(1, num_samples, dtype=torch.float32, device=module.device)

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
    extra_files = {key: value.encode("utf-8") for key, value in metadata.items()}

    # Use the internal name expected by this PyTorch build for extra files
    torch.jit.save(scripted, str(output_path), _extra_files=extra_files)

    print(f"TorchScript module saved to: {output_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
