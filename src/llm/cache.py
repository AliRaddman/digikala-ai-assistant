"""Deterministic SQLite cache for validated LLM responses.

Owner: Benyamin. Cache keys contain SHA-256 hashes, not raw prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from array import array
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


@dataclass(frozen=True, slots=True)
class SemanticCacheMatch:
    entry: CacheEntry
    similarity: float
    source_cache_key: str


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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_semantic_cache (
                    guard_key TEXT NOT NULL,
                    source_cache_key TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    vector_dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (guard_key, source_cache_key, embedding_model),
                    FOREIGN KEY (source_cache_key)
                        REFERENCES llm_cache(cache_key)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_semantic_guard
                ON llm_semantic_cache (guard_key, embedding_model, created_at)
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

    def get_semantic(
        self,
        *,
        guard_key: str,
        embedding_model: str,
        query_vector: list[float],
        threshold: float,
        max_candidates: int = 256,
    ) -> SemanticCacheMatch | None:
        """Return the most similar non-expired entry within one exact guard."""
        if not 0 <= threshold <= 1:
            raise ValueError("semantic cache threshold must be between 0 and 1")
        query = _normalise_vector(query_vector)
        if not query:
            return None

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.source_cache_key, s.vector_blob, s.vector_dimension,
                       c.model, c.response_json, c.usage_json,
                       c.original_cost_usd, c.created_at
                FROM llm_semantic_cache AS s
                JOIN llm_cache AS c ON c.cache_key = s.source_cache_key
                WHERE s.guard_key = ? AND s.embedding_model = ?
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (guard_key, embedding_model, max_candidates),
            ).fetchall()

        now = datetime.now(UTC)
        best: SemanticCacheMatch | None = None
        for row in rows:
            created_at = datetime.fromisoformat(row[7])
            if self.ttl is not None and now - created_at > self.ttl:
                continue
            dimension = int(row[2])
            if dimension != len(query):
                continue
            stored_array = array("f")
            stored_array.frombytes(row[1])
            if len(stored_array) != dimension:
                continue
            similarity = sum(
                left * right for left, right in zip(query, stored_array, strict=True)
            )
            if similarity < threshold:
                continue
            entry = CacheEntry(
                data=json.loads(row[4]),
                model=row[3],
                usage=TokenUsage.from_dict(json.loads(row[5])),
                original_cost_usd=float(row[6]),
                created_at=created_at,
            )
            if best is None or similarity > best.similarity:
                best = SemanticCacheMatch(
                    entry=entry,
                    similarity=float(similarity),
                    source_cache_key=row[0],
                )
        return best

    def set_semantic(
        self,
        *,
        guard_key: str,
        source_cache_key: str,
        embedding_model: str,
        vector: list[float],
    ) -> None:
        normalised = _normalise_vector(vector)
        if not normalised:
            raise ValueError("semantic cache vector cannot be empty or zero")
        vector_array = array("f", normalised)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_semantic_cache (
                    guard_key, source_cache_key, embedding_model,
                    vector_blob, vector_dimension, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(guard_key, source_cache_key, embedding_model)
                DO UPDATE SET
                    vector_blob = excluded.vector_blob,
                    vector_dimension = excluded.vector_dimension,
                    created_at = excluded.created_at
                """,
                (
                    guard_key,
                    source_cache_key,
                    embedding_model,
                    sqlite3.Binary(vector_array.tobytes()),
                    len(vector_array),
                    now,
                ),
            )


def _normalise_vector(vector: list[float]) -> list[float]:
    values = [float(value) for value in vector]
    magnitude = math.sqrt(sum(value * value for value in values))
    if not values or magnitude == 0 or not math.isfinite(magnitude):
        return []
    normalised = [value / magnitude for value in values]
    if not all(math.isfinite(value) for value in normalised):
        return []
    return normalised
