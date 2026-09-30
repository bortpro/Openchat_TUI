"""Chat domain types plus JSON persistence for threads and settings."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import config

NEW_CHAT_TITLE = "New chat"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str = ""
    reasoning: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ChatThread:
    title: str = NEW_CHAT_TITLE
    messages: list[ChatMessage] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def touch(self) -> None:
        self.updated_at = _now()

    @classmethod
    def from_dict(cls, d: dict) -> "ChatThread":
        return cls(
            title=d.get("title", NEW_CHAT_TITLE),
            messages=[
                ChatMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    reasoning=m.get("reasoning", ""),
                    id=m.get("id", str(uuid.uuid4())),
                )
                for m in d.get("messages", [])
            ],
            id=d.get("id", str(uuid.uuid4())),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
        )


def title_preview(text: str) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= 36:
        return t or NEW_CHAT_TITLE
    return t[:36] + "…"


def estimate_tokens(texts: list[str]) -> int:
    """Rough 0.75 x words estimate, matching the iOS TokenEstimator."""
    words = sum(len(t.split()) for t in texts)
    return math.ceil(0.75 * words)


def thread_tokens(thread: ChatThread, draft: str = "") -> int:
    texts = [draft]
    for m in thread.messages:
        texts += [m.content, m.reasoning]
    return estimate_tokens(texts)


def _write_json(path, data) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def load_threads() -> list[ChatThread]:
    try:
        raw = json.loads(config.THREADS_FILE.read_text())
        return [ChatThread.from_dict(d) for d in raw]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return []


def save_threads(threads: list[ChatThread]) -> None:
    _write_json(config.THREADS_FILE, [asdict(t) for t in threads])


@dataclass
class Settings:
    model_id: str = config.DEFAULT_MODEL.id
    thinking_enabled: bool = True

    @property
    def model(self) -> config.ChatModel:
        return config.model_by_id(self.model_id)


def load_settings() -> Settings:
    try:
        raw = json.loads(config.SETTINGS_FILE.read_text())
        return Settings(
            model_id=config.model_by_id(raw.get("model_id")).id,
            thinking_enabled=bool(raw.get("thinking_enabled", True)),
        )
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return Settings()


def save_settings(settings: Settings) -> None:
    _write_json(config.SETTINGS_FILE, asdict(settings))
