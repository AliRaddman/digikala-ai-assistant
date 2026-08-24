"""Token, latency and API-cost accounting.

Owner: Benyamin. The ledger intentionally stores no prompt or response text.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
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
                    saved_cost_usd REAL NOT NULL
                )
                """
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
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_usage (
                    created_at, operation, model, request_id, cache_hit,
                    input_tokens, cached_input_tokens, output_tokens,
                    latency_ms, cost_usd, saved_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS logical_requests,
                       COALESCE(SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END), 0)
                           AS api_calls,
                       COALESCE(SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END), 0)
                           AS cache_hits,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd,
                       COALESCE(SUM(saved_cost_usd), 0.0) AS saved_cost_usd,
                       COALESCE(AVG(latency_ms), 0.0) AS mean_latency_ms
                FROM llm_usage
                """
            ).fetchone()
        return dict(row)
