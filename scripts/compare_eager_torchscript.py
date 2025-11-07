#!/usr/bin/env python3
"""Compare eager vs TorchScript embeddings for one or more audio files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the finetuned checkpoint (model_lmft_0.14.pth).")
    parser.add_argument("--audio", type=Path, nargs="+", help="Audio files to evaluate. If omitted, compares a synthetic random waveform.")
    parser.add_argument("--model-path", type=Path, required=True, help="Directory containing model.safetensors and config.json.")
    parser.add_argument("--scripted", type=Path, required=True, help="TorchScript artifact to load (waveform-based).")
    parser.add_argument("--preprocessed", type=Path, default=None, help="Optional preprocessed TorchScript artifact (features in).")
    parser.add_argument("--device", default="cpu", help="Device for eager model (default cpu).")
    return parser.parse_args()


def resolve_audio_paths(repo_root: Path, provided: Iterable[Path]) -> list[Path]:
    if provided:
        paths = []
        for path in provided:
            resolved = path if path.is_absolute() else (repo_root / path).resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            paths.append(resolved)
        return paths
    # fallback: use the demo sample
    candidate = (repo_root / "../datasets/voxceleb1test/wav/id10270/5r0dWxy17C8/00001.wav").resolve()
    if not candidate.exists():
        raise FileNotFoundError("Provide at least one audio file via --audio")
    return [candidate]


def load_waveform(path: Path, target_sr: int) -> torch.Tensor:
    sig, sr = sf.read(str(path), dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != target_sr:
        import librosa

        sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(sig).unsqueeze(0).to(torch.float32)


def preload_feature_extractor(mod):
    spk = mod.modules_dict["spk_model"]
    front = spk.front
    return front, front.feature_extractor, spk.n_mfa_layers


def compute_embeddings_eager(mod, waveform: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        emb = mod(waveform.to(mod.device))
    return emb.squeeze(0).detach().cpu()


def compute_embeddings_scripted(scripted, waveform: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        emb = scripted(waveform.to(scripted.device if hasattr(scripted, "device") else "cpu"))
    return emb.squeeze(0).detach().cpu()


def compute_features(front, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    array = waveform.squeeze(0).cpu().numpy()
    features = front.feature_extractor(
        array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    input_features = features["input_features"].to(dtype=next(front.encoder.parameters()).dtype)
    return input_features


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    # Load eager model
    from w2vbert_speaker.module import W2VBERT_SPK_Module

    eager = W2VBERT_SPK_Module(device=args.device, model_path=str(args.model_path))
    eager.load_model(args.checkpoint)
    eager.eval()
    target_sr = eager.modules_dict["spk_model"].front.feature_extractor.sampling_rate

    # Load scripted models
    scripted = torch.jit.load(str(args.scripted), map_location="cpu")
    preprocessed = None
    if args.preprocessed is not None:
        preprocessed = torch.jit.load(str(args.preprocessed), map_location="cpu")

    audio_paths = resolve_audio_paths(repo_root, args.audio)
    print(f"Comparing embeddings for {len(audio_paths)} audio file(s)")

    front, feature_extractor, n_mfa_layers = preload_feature_extractor(eager)

    for path in audio_paths:
        waveform = load_waveform(path, target_sr)
        eager_emb = compute_embeddings_eager(eager, waveform)
        scripted_emb = compute_embeddings_scripted(scripted, waveform)

        sim = torch.nn.functional.cosine_similarity(eager_emb.unsqueeze(0), scripted_emb.unsqueeze(0)).item()
        l2 = torch.norm(eager_emb - scripted_emb).item()

        print(f"\nFile: {path}")
        print(f"  cosine(eager, scripted) = {sim:.6f}")
        print(f"  L2(eager, scripted) = {l2:.6e}")

        if preprocessed is not None:
            features = compute_features(front, waveform, target_sr)
            with torch.inference_mode():
                pre_emb = preprocessed(features)
            pre_emb = pre_emb.squeeze(0).detach().cpu()
            sim_pre = torch.nn.functional.cosine_similarity(eager_emb.unsqueeze(0), pre_emb.unsqueeze(0)).item()
            l2_pre = torch.norm(eager_emb - pre_emb).item()
            print(f"  cosine(eager, preprocessed) = {sim_pre:.6f}")
            print(f"  L2(eager, preprocessed) = {l2_pre:.6e}")


if __name__ == "__main__":
    main()
