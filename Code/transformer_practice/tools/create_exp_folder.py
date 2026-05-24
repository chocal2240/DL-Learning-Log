"""Create numbered experiment folders under ``run/``."""

from __future__ import annotations

from pathlib import Path

import config


def _create_numbered_folder(parent: Path, base_name: str) -> Path:
    """Create ``base_name`` or the next available ``base_nameN`` folder."""

    parent.mkdir(parents=True, exist_ok=True)

    for index in range(0, 10_000):
        folder_name = base_name if index == 0 else f"{base_name}{index}"
        folder = parent / folder_name
        if not folder.exists():
            folder.mkdir()
            return folder

    raise RuntimeError(f"Could not create a new experiment folder under {parent}")


def create_exp_folder() -> tuple[Path, Path]:
    """Create a training experiment folder and its weights subfolder."""

    exp_folder = _create_numbered_folder(config.PROJECT_ROOT / "run" / "train", "exp")
    weights_folder = exp_folder / "weights"
    weights_folder.mkdir()
    return exp_folder, weights_folder


def create_val_exp_folder() -> Path:
    """Create a prediction experiment folder."""

    return _create_numbered_folder(config.PROJECT_ROOT / "run" / "predict", "exp")
