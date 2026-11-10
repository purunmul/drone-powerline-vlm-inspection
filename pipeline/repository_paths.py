"""Resolve original experiment image paths after moving the checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path('/home/work/123-0125')


def image_path(value):
    path = Path(value)
    if path.is_relative_to(ORIGINAL_ROOT):
        return ROOT / path.relative_to(ORIGINAL_ROOT)
    return path if path.is_absolute() else ROOT / path
