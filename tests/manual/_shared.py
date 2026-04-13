"""Shared helpers for ad-hoc manual test scripts."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_project_root_to_syspath() -> Path:
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return PROJECT_ROOT


def get_env_file() -> Path:
    return PROJECT_ROOT / "config" / ".env"


def get_sycophancy_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "external" / "sycophancy_database"


def get_samples_dir() -> Path:
    return PROJECT_ROOT / "data" / "samples"
