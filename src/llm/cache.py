"""Deterministic SQLite cache for validated LLM responses.

Owner: Benyamin. Cache keys contain SHA-256 hashes, not raw prompts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.llm.types import TokenUsage


@dataclass(frozen=True, slots=True)
class CacheEntry:
    data: dict[str, Any]
    model: str
    usage: TokenUsage
    original_cost_usd: float
    created_at: datetime


class SQLiteLLMCache:
    """A small process-safe disk cache backed by SQLite."""

    def __init__(self, path: Path, ttl: timedelta | None = None) -> None:
        self.path = path
        self.ttl = ttl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    original_cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> CacheEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT model, response_json, usage_json,
                       original_cost_usd, created_at
                FROM llm_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None

        created_at = datetime.fromisoformat(row[4])
        if self.ttl is not None and datetime.now(UTC) - created_at > self.ttl:
            return None
        return CacheEntry(
            data=json.loads(row[1]),
            model=row[0],
            usage=TokenUsage.from_dict(json.loads(row[2])),
            original_cost_usd=float(row[3]),
            created_at=created_at,
        )

    def set(
        self,
        cache_key: str,
        *,
        data: dict[str, Any],
        model: str,
        usage: TokenUsage,
        original_cost_usd: float,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_cache (
                    cache_key, model, response_json, usage_json,
                    original_cost_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model = excluded.model,
                    response_json = excluded.response_json,
                    usage_json = excluded.usage_json,
                    original_cost_usd = excluded.original_cost_usd,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    model,
                    json.dumps(data, ensure_ascii=False, sort_keys=True),
                    json.dumps(usage.as_dict(), sort_keys=True),
                    original_cost_usd,
                    now,
                ),
            )
