"""Token, latency and API-cost accounting.

Owner: Benyamin. The ledger intentionally stores no prompt or response text.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil, floor
from pathlib import Path
from typing import Any

from src.llm.types import TokenUsage


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


_KNOWN_PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(
        input_per_million=0.15,
        cached_input_per_million=0.075,
        output_per_million=0.60,
    )
}


def pricing_for_model(model: str) -> ModelPricing | None:
    if model in _KNOWN_PRICING:
        return _KNOWN_PRICING[model]
    for alias, pricing in _KNOWN_PRICING.items():
        if model.startswith(f"{alias}-"):
            return pricing
    return None


def estimate_cost_usd(model: str, usage: TokenUsage) -> float:
    pricing = pricing_for_model(model)
    if pricing is None:
        raise ValueError(
            f"no pricing configured for model {model!r}; refusing to log a false "
            "zero cost"
        )
    cost = (
        usage.uncached_input_tokens * pricing.input_per_million
        + usage.cached_input_tokens * pricing.cached_input_per_million
        + usage.output_tokens * pricing.output_per_million
    ) / 1_000_000
    return round(cost, 12)


class SQLiteUsageLedger:
    """Append-only per-call metrics suitable for budget and latency reports."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    request_id TEXT,
                    cache_hit INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    saved_cost_usd REAL NOT NULL,
                    cache_type TEXT NOT NULL DEFAULT 'none',
                    cache_similarity REAL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(llm_usage)")
            }
            if "cache_type" not in columns:
                connection.execute(
                    "ALTER TABLE llm_usage "
                    "ADD COLUMN cache_type TEXT NOT NULL DEFAULT 'none'"
                )
                connection.execute(
                    "UPDATE llm_usage SET cache_type = 'exact' WHERE cache_hit = 1"
                )
            if "cache_similarity" not in columns:
                connection.execute(
                    "ALTER TABLE llm_usage ADD COLUMN cache_similarity REAL"
                )

    def record(
        self,
        *,
        operation: str,
        model: str,
        request_id: str | None,
        cache_hit: bool,
        usage: TokenUsage,
        latency_ms: float,
        cost_usd: float,
        saved_cost_usd: float = 0.0,
        cache_type: str | None = None,
        cache_similarity: float | None = None,
    ) -> None:
        resolved_cache_type = cache_type or ("exact" if cache_hit else "none")
        if resolved_cache_type not in {"none", "exact", "semantic"}:
            raise ValueError(f"invalid cache_type: {resolved_cache_type!r}")
        if cache_hit != (resolved_cache_type != "none"):
            raise ValueError("cache_hit and cache_type disagree")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_usage (
                    created_at, operation, model, request_id, cache_hit,
                    input_tokens, cached_input_tokens, output_tokens,
                    latency_ms, cost_usd, saved_cost_usd, cache_type,
                    cache_similarity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    operation,
                    model,
                    request_id,
                    int(cache_hit),
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    latency_ms,
                    cost_usd,
                    saved_cost_usd,
                    resolved_cache_type,
                    cache_similarity,
                ),
            )

    def checkpoint(self) -> int:
        """Return a stable row id for measuring one later evaluation run."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) AS last_id FROM llm_usage"
            ).fetchone()
        return int(row["last_id"])

    def summary(
        self,
        *,
        after_id: int = 0,
        operation: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate usage after a checkpoint, optionally for one operation.

        Percentiles are calculated from individual logical-request latencies,
        so cache hits remain visible instead of being mixed into an API-only
        timing number.
        """

        where = ["id > ?"]
        params: list[Any] = [after_id]
        if operation is not None:
            where.append("operation = ?")
            params.append(operation)
        where_sql = " AND ".join(where)

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS logical_requests,
                       COALESCE(SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END), 0)
                           AS api_calls,
                       COALESCE(SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END), 0)
                           AS cache_hits,
                       COALESCE(SUM(CASE WHEN cache_type = 'exact' THEN 1 ELSE 0 END), 0)
                           AS exact_cache_hits,
                       COALESCE(SUM(CASE WHEN cache_type = 'semantic' THEN 1 ELSE 0 END), 0)
                           AS semantic_cache_hits,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                       COALESCE(SUM(saved_cost_usd), 0.0) AS saved_cost_usd,
                       COALESCE(AVG(latency_ms), 0.0) AS mean_latency_ms,
                       AVG(CASE WHEN cache_type = 'semantic'
                                THEN cache_similarity END)
                           AS mean_semantic_similarity
                FROM llm_usage
                WHERE {where_sql}
                """,
                params,
            ).fetchone()
            latency_rows = connection.execute(
                f"""
                SELECT latency_ms
                FROM llm_usage
                WHERE {where_sql}
                ORDER BY latency_ms
                """,
                params,
            ).fetchall()

        summary = dict(row)
        latencies = [float(item["latency_ms"]) for item in latency_rows]
        requests = int(summary["logical_requests"])
        summary["cache_hit_rate"] = (
            float(summary["cache_hits"]) / requests if requests else 0.0
        )
        summary["p50_latency_ms"] = _percentile(latencies, 50)
        summary["p95_latency_ms"] = _percentile(latencies, 95)
        return summary


def _percentile(values: list[float], percentile: float) -> float:
    """Linearly interpolated percentile; returns zero for an empty series."""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
