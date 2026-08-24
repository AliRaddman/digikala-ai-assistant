"""Environment-backed LLM configuration.

Owner: Benyamin. Secrets are read at runtime and are never written to logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LLMSettings:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    cache_path: Path = Path("data/cache/llm_cache.sqlite3")
    usage_path: Path = Path("data/cache/llm_usage.sqlite3")
    timeout_seconds: float = 30.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            backend=os.getenv("LLM_BACKEND", "openai").strip().lower(),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") or None,
            cache_path=Path(
                os.getenv("LLM_CACHE_PATH", "data/cache/llm_cache.sqlite3")
            ),
            usage_path=Path(
                os.getenv("LLM_USAGE_PATH", "data/cache/llm_usage.sqlite3")
            ),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY or OPENAI_API_KEY is required for live LLM calls"
            )
        return self.api_key
