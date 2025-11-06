#!/usr/bin/env python3
"""Create a virtual environment for running the w2v-BERT embedding notebook.

The script creates a self-contained Python virtual environment, installs the
packages required to run `W2VBERT_SPK_Module` for embedding extraction, and
registers the environment as a Jupyter kernel so it can be selected directly
inside notebooks.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_VENV_DIR = Path(".venv-w2vbert-notebook")
DEFAULT_KERNEL_NAME = "w2vbert-notebook"
REQUIRED_PACKAGES = [
    "torch",
    "torchaudio",
    "transformers>=4.45",
    "accelerate",
    "huggingface-hub",
    "sentencepiece",
    "soundfile",
    "librosa",
    "hyperpyyaml",
    "speechbrain",
    "peft",
    "numpy",
    "scipy",
    "ipykernel",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_VENV_DIR,
        help="Directory for the virtual environment (relative paths are resolved from the repo root).",
    )
    parser.add_argument(
        "--kernel-name",
        default=DEFAULT_KERNEL_NAME,
        help="Name to register for the Jupyter kernel.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Optional path to the Python executable used to create the virtual environment.",
    )
    parser.add_argument(
        "--display-name",
        default="w2v-BERT Notebook",
        help="Display name shown inside Jupyter clients for the registered kernel.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the virtual environment even if it already exists.",
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Skip Jupyter kernel registration (useful in CI environments).",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "recipes").exists() and (candidate / "deeplab").exists():
            return candidate
    raise RuntimeError("Unable to locate the repository root from this script path.")


def ensure_environment(venv_path: Path, recreate: bool, python: str | None) -> None:
    if venv_path.exists() and not recreate:
        print(f"Virtual environment already exists at {venv_path}. Skipping creation.")
        return
    if venv_path.exists() and recreate:
        print(f"Removing existing virtual environment at {venv_path}.")
        shutil.rmtree(venv_path)
    print(f"Creating virtual environment at {venv_path}.")
    creator = python if python is not None else sys.executable
    run_with_output([creator, "-m", "venv", str(venv_path)])


def python_executable(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run_with_output(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.check_call(command, env=env)


def install_packages(python_path: Path) -> None:
    run_with_output([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run_with_output([str(python_path), "-m", "pip", "install", *REQUIRED_PACKAGES])


def register_kernel(python_path: Path, kernel_name: str, display_name: str) -> None:
    run_with_output(
        [
            str(python_path),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            kernel_name,
            "--display-name",
            display_name,
        ]
    )


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    venv_path = (repo_root / args.venv_dir).resolve()

    ensure_environment(venv_path, args.recreate, args.python)

    python_path = python_executable(venv_path)
    if not python_path.exists():
        raise FileNotFoundError(f"Virtual environment Python not found at {python_path}")

    install_packages(python_path)

    if not args.skip_kernel:
        register_kernel(python_path, args.kernel_name, args.display_name)

    print(
        "\nSetup complete. You can now select the kernel "
        f"'{args.display_name}' inside Jupyter to run the notebook."
    )


if __name__ == "__main__":
    main()
