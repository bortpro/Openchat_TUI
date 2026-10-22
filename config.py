"""Static configuration for openchat TUI: API key, NIM models, and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Put your NVIDIA NIM API key in local_config.py (gitignored) as
# NVIDIA_API_KEY = "nvapi-...". If that file is missing or the key is empty,
# the NVIDIA_API_KEY environment variable is used instead.
try:
    from local_config import NVIDIA_API_KEY
except ImportError:
    NVIDIA_API_KEY = "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"

BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

INSTANT_MODE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and directly. Do not output "
    "chain-of-thought, scratch reasoning, or “thinking” sections—only the reply "
    "the user should read."
)

TEMPERATURE = 1
TOP_P = 0.95
MAX_TOKENS = 16384

# Soft client-side limit, same as the iOS app's TokenEstimator.threadLimit.
THREAD_TOKEN_LIMIT = 200_000

DATA_DIR = Path(os.environ.get("OPENCHAT_TUI_HOME", Path.home() / ".openchat_tui"))
THREADS_FILE = DATA_DIR / "threads.json"
SETTINGS_FILE = DATA_DIR / "settings.json"


@dataclass(frozen=True)
class ChatModel:
    id: str
    title: str


MODELS: list[ChatModel] = [
    ChatModel("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B A55B"),
    ChatModel("nvidia/nemotron-3.5-super-vl-120b-a12b", "Nemotron 3.5 Super VL 120B A12B"),
]
DEFAULT_MODEL = MODELS[0]


def model_by_id(model_id: str | None) -> ChatModel:
    return next((m for m in MODELS if m.id == model_id), DEFAULT_MODEL)


def api_key() -> str:
    return (NVIDIA_API_KEY or os.environ.get("NVIDIA_API_KEY", "")).strip()
