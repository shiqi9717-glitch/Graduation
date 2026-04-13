"""Environment and API-key helpers shared across modules."""

import os
from pathlib import Path

from dotenv import load_dotenv


PLACEHOLDER_KEYS = {
    "",
    "your_api_key_here",
    "your_deepseek_api_key_here",
    "change_me",
    "replace_me",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_env_path() -> Path:
    return get_project_root() / "config" / ".env"


def load_project_env() -> Path:
    env_path = get_env_path()
    load_dotenv(env_path)
    return env_path


def to_bool(value: str, default: bool = False) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "y", "on"}


def is_placeholder_key(value: str) -> bool:
    return str(value or "").strip().lower() in PLACEHOLDER_KEYS


def get_env_flag(name: str, default: bool = False) -> bool:
    return to_bool(os.getenv(name, ""), default=default)
