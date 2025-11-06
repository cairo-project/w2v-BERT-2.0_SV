#!/usr/bin/env python3
"""Generate a W2V-BERT speaker embedding for the demo audio sample.

The script mirrors the notebook walk-through so that the pipeline can be
validated quickly from the command line. The target audio path can be
customised via ``--audio`` if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch


DEMO_AUDIO = Path("../datasets/voxceleb1test/wav/id10270/5r0dWxy17C8/00001.wav")


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "recipes").exists() and (candidate / "deeplab").exists():
            return candidate
    raise RuntimeError("Unable to locate repository root. Run the script from inside the repo.")


def extend_sys_path(repo_root: Path) -> None:
    additions = [
        repo_root,
        repo_root / "recipes/DeepASV",
        repo_root / "deeplab/pretrained/audio2vector/module/transformers/src",
    ]
    for candidate in reversed(additions):
        if candidate.exists():
            resolved = str(candidate)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)


def ensure_checkpoint(
    repo_root: Path, override: Path | None, candidates: tuple[Path, ...]
) -> Path:
    if override is not None:
        resolved = override if override.is_absolute() else (repo_root / override).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resolved}")
        return resolved

    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected_locations = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "No checkpoint found. Provide --checkpoint or place the weights at one of these paths:\n"
        f"{expected_locations}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEMO_AUDIO,
        help="Audio file (wav) used for the embedding demo.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device specifier passed to W2VBERT_SPK_Module (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path overriding the default provided by the module.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional configuration file overriding the default provided by the module.",
    )
    return parser.parse_args()


def load_waveform(audio_path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    signal, sr = sf.read(str(audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    waveform = torch.from_numpy(signal).unsqueeze(0).to(torch.float32)
    return waveform, sr


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    extend_sys_path(repo_root)

    from recipes.DeepASV.utils.inference import (
        W2VBERT_SPK_Module,
        get_checkpoint_candidates,
    )

    config = args.config if args.config is not None else None
    candidates = get_checkpoint_candidates()
    checkpoint = ensure_checkpoint(repo_root, args.checkpoint, candidates)

    module_kwargs = {}
    if args.device is not None:
        module_kwargs["device"] = args.device
    if config is not None:
        module_kwargs["path"] = config

    model = W2VBERT_SPK_Module(**module_kwargs)
    model.load_model(ckpt_path=checkpoint)

    audio_path = args.audio
    if not audio_path.is_absolute():
        audio_path = (repo_root / audio_path).resolve()
    if not audio_path.exists():
        # Try the parent repo relative path if running from inside repo_root.
        fallback = (repo_root.parent / args.audio).resolve()
        if fallback.exists():
            audio_path = fallback
        else:
            raise FileNotFoundError(f"Audio file not found: {args.audio}")

    target_sr = model.hparams.get("sample_rate", 16000)
    waveform, sr = load_waveform(audio_path, target_sr)
    print(f"Loaded {audio_path} (sample rate {sr}, waveform shape {waveform.shape}).")

    with torch.inference_mode():
        embedding = model(waveform)

    vector = embedding.squeeze(0).detach().cpu()
    print(f"Embedding shape: {vector.shape}")
    preview = vector[:8].tolist()
    print("First 8 values:", " ".join(f"{value:.4f}" for value in preview))


if __name__ == "__main__":
    main()
