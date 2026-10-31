"""Shared input/output paths; importing this module does not create directories."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LYMPHOMA_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("LYMPHOMA_OUTPUT_DIR", PROJECT_ROOT / "results")).expanduser().resolve()


def data_path(relative):
    return str(DATA_DIR / relative)


def output_path(relative):
    return str(OUTPUT_DIR / relative)
