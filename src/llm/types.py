"""Shared LLM value objects.

Owner: Benyamin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts returned by the model provider for one billed request."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TokenUsage":
        return cls(
            input_tokens=int(value.get("input_tokens", 0)),
            cached_input_tokens=int(value.get("cached_input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
        )


@dataclass(frozen=True, slots=True)
class StructuredResult:
    """Validated JSON plus billing metadata for one logical LLM request."""

    data: dict[str, Any]
    model: str
    request_id: str | None
    usage: TokenUsage
    latency_ms: float
    cache_hit: bool
    cost_usd: float
    saved_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "model": self.model,
            "request_id": self.request_id,
            "usage": self.usage.as_dict(),
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "cost_usd": self.cost_usd,
            "saved_cost_usd": self.saved_cost_usd,
        }
