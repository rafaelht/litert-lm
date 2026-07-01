from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


@dataclass(frozen=True)
class Settings:
    model_path: str
    server_port: int
    session_timeout: int
    max_active_conversations: int

    @property
    def model_id(self) -> str:
        return Path(self.model_path).parent.name or "litert-model"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    raw_model_path = os.getenv("MODEL_PATH", "/models/gemma-4-E2B-it.litertlm/model.litertlm")
    model_path = raw_model_path.strip()

    if not model_path.startswith("/") and not model_path.startswith("\\") and not model_path.startswith("~"):
        model_path = str((project_root / model_path).resolve())

    return Settings(
        model_path=model_path,
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        session_timeout=int(os.getenv("SESSION_TIMEOUT", "1800")),
        max_active_conversations=int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "1000")),
    )
