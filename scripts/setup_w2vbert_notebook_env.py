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
from typing import Iterable, List


DEFAULT_VENV_DIR = Path(".venv-w2vbert-notebook")
DEFAULT_KERNEL_NAME = "w2vbert-notebook"
DEFAULT_REQUIREMENTS = Path("requirements.txt")
MANDATORY_PACKAGES = [
    "librosa",
]
OVERRIDDEN_PACKAGES = {
    "numpy": "numpy<2",
}
UNINSTALL_AFTER_SETUP = [
    "transformers",
]

SKIP_PACKAGE_NAMES = {
    "transformers",  # Provided as source inside the repository
    "pip",
    "setuptools",
    "wheel",
    "triton",
}
SKIP_PACKAGE_PREFIXES = (
    "nvidia-",
    "cuda-",
)
TORCH_FAMILY = {"torch", "torchaudio", "torchvision", "triton"}


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
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Path to the requirements file to install.",
    )
    parser.add_argument(
        "--extra-package",
        action="append",
        default=[],
        help="Additional package specifiers to install after requirements.",
    )
    parser.add_argument(
        "--torch-index-url",
        default=None,
        help="Optional custom index URL for installing PyTorch wheels.",
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "recipes").exists() and (candidate / "deeplab").exists():
            return candidate
    raise RuntimeError("Unable to locate the repository root from this script path.")


def resolve_python_executable(requested_python: str | None) -> str:
    if requested_python:
        return str(Path(requested_python))

    current = Path(sys.executable)
    version_info = sys.version_info
    if version_info >= (3, 13):
        for candidate in ("python3.12", "python3.11", "python3.10"):
            resolved = shutil.which(candidate)
            if resolved:
                print(
                    f"Selecting {resolved} for virtualenv creation to maintain PyTorch compatibility (current interpreter is Python {version_info.major}.{version_info.minor})."
                )
                return resolved
        print(
            "Warning: Python 3.13+ detected and no fallback interpreter found. PyTorch installation may fail; consider providing --python."
        )
    return str(current)


def ensure_environment(venv_path: Path, recreate: bool, python: str | None) -> Path:
    if venv_path.exists() and not recreate:
        print(f"Virtual environment already exists at {venv_path}. Skipping creation.")
        return python_executable(venv_path)
    if venv_path.exists() and recreate:
        print(f"Removing existing virtual environment at {venv_path}.")
        shutil.rmtree(venv_path)
    print(f"Creating virtual environment at {venv_path}.")
    creator = resolve_python_executable(python)
    run_with_output([creator, "-m", "venv", str(venv_path)])
    return python_executable(venv_path)


def python_executable(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run_with_output(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.check_call(command, env=env)


def parse_requirements(requirements_path: Path) -> List[str]:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    parsed: List[str] = []
    seen_names: set[str] = set()
    with requirements_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("package") or line.startswith("-"):
                continue
            if line == "pip-requirements":
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            name, version = parts[0], parts[1]

            if name in SKIP_PACKAGE_NAMES:
                continue
            if any(name.startswith(prefix) for prefix in SKIP_PACKAGE_PREFIXES):
                continue
            if version.lower() in {"version", "-"}:
                continue

            if name in OVERRIDDEN_PACKAGES:
                override = OVERRIDDEN_PACKAGES[name]
                if override not in parsed:
                    parsed.append(override)
                    seen_names.add(name)
                continue

            spec = f"{name}=={version}"
            if name not in seen_names:
                parsed.append(spec)
                seen_names.add(name)

    for override_name, override_spec in OVERRIDDEN_PACKAGES.items():
        if override_name not in seen_names:
            parsed.append(override_spec)
            seen_names.add(override_name)
    return parsed


def split_torch_packages(packages: Iterable[str]) -> tuple[list[str], list[str]]:
    torch_packages: list[str] = []
    other_packages: list[str] = []
    for package in packages:
        name = package.split("==")[0]
        if name in TORCH_FAMILY:
            torch_packages.append(name)
        else:
            other_packages.append(package)
    return torch_packages, other_packages


def install_packages(
    python_path: Path,
    packages: List[str],
    torch_index_url: str | None,
    extra_packages: list[str],
) -> None:
    run_with_output([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

    requested = packages + MANDATORY_PACKAGES + extra_packages

    deduped_requested: list[str] = []
    seen: set[str] = set()
    for spec in requested:
        if spec not in seen:
            deduped_requested.append(spec)
            seen.add(spec)

    torch_packages, other_packages = split_torch_packages(deduped_requested)

    if other_packages:
        run_with_output([str(python_path), "-m", "pip", "install", *other_packages])

    if torch_packages:
        unique_packages = sorted(set(torch_packages))
        print(
            "Installing PyTorch family packages without pinned versions: "
            + ", ".join(unique_packages)
        )
        command = [str(python_path), "-m", "pip", "install", *unique_packages]
        if torch_index_url:
            command.extend(["--index-url", torch_index_url])
        try:
            run_with_output(command)
        except subprocess.CalledProcessError as error:
            print(
                "Warning: PyTorch installation failed. The environment may require manual setup."
            )
            raise error

    for package in UNINSTALL_AFTER_SETUP:
        print(
            f"Ensuring '{package}' wheel is removed so the in-repo version is used when present."
        )
        subprocess.run(
            [str(python_path), "-m", "pip", "uninstall", "-y", package],
            check=False,
        )


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

    python_path = ensure_environment(venv_path, args.recreate, args.python)

    if not python_path.exists():
        raise FileNotFoundError(f"Virtual environment Python not found at {python_path}")

    version_output = subprocess.check_output(
        [str(python_path), "--version"], text=True
    ).strip()
    print(f"Environment interpreter: {version_output}")

    if " 3.13" in version_output or " 3.14" in version_output:
        raise RuntimeError(
            "PyTorch wheels are unavailable for Python >=3.13. Re-run the script with --python pointing to a 3.12/3.11 interpreter."
        )

    requirements_path = (
        args.requirements
        if args.requirements.is_absolute()
        else (repo_root / args.requirements)
    )
    requirements = parse_requirements(requirements_path)

    install_packages(python_path, requirements, args.torch_index_url, args.extra_package)

    if not args.skip_kernel:
        register_kernel(python_path, args.kernel_name, args.display_name)

    print(
        "\nSetup complete. You can now select the kernel "
        f"'{args.display_name}' inside Jupyter to run the notebook."
    )


if __name__ == "__main__":
    main()
