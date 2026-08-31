"""Cached structured-output LLM client and OpenAI Responses provider.

Owner: Benyamin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from src.llm.cache import SQLiteLLMCache
from src.llm.config import LLMSettings
from src.llm.semantic_cache import (
    SemanticCacheRequest,
    SemanticEncoder,
    SentenceTransformerSemanticEncoder,
)
from src.llm.types import StructuredResult, TokenUsage
from src.llm.usage import SQLiteUsageLedger, estimate_cost_usd

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ProviderResult:
    data: dict[str, Any]
    model: str
    request_id: str | None
    usage: TokenUsage


class StructuredLLMProvider(Protocol):
    def generate_structured(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type[SchemaT],
    ) -> ProviderResult: ...


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter using SDK-native Pydantic parsing."""

    def __init__(self, settings: LLMSettings) -> None:
        if settings.backend != "openai":
            raise ValueError(f"unsupported LLM_BACKEND: {settings.backend!r}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is missing; run pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": settings.require_api_key(),
            "timeout": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = OpenAI(**kwargs)

    def generate_structured(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type[SchemaT],
    ) -> ProviderResult:
        response = self._client.responses.parse(
            model=model,
            input=messages,
            text_format=response_model,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("model returned no parsed structured output")
        validated = (
            parsed
            if isinstance(parsed, response_model)
            else response_model.model_validate(parsed)
        )

        usage_object = getattr(response, "usage", None)
        input_details = getattr(usage_object, "input_tokens_details", None)
        usage = TokenUsage(
            input_tokens=int(getattr(usage_object, "input_tokens", 0) or 0),
            cached_input_tokens=int(
                getattr(input_details, "cached_tokens", 0) or 0
            ),
            output_tokens=int(getattr(usage_object, "output_tokens", 0) or 0),
        )
        return ProviderResult(
            data=validated.model_dump(mode="json"),
            model=str(getattr(response, "model", None) or model),
            request_id=getattr(response, "id", None),
            usage=usage,
        )


class CachedLLMClient:
    """Adds deterministic cache and usage accounting around any provider."""

    def __init__(
        self,
        *,
        provider: StructuredLLMProvider,
        model: str,
        cache: SQLiteLLMCache,
        ledger: SQLiteUsageLedger,
        semantic_encoder: SemanticEncoder | None = None,
        semantic_threshold: float = 0.96,
    ) -> None:
        if not 0 <= semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be between 0 and 1")
        self.provider = provider
        self.model = model
        self.cache = cache
        self.ledger = ledger
        self.semantic_encoder = semantic_encoder
        self.semantic_threshold = semantic_threshold

    def generate_structured(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]],
        response_model: type[SchemaT],
        cache_namespace: str,
        semantic_cache: SemanticCacheRequest | None = None,
    ) -> StructuredResult:
        response_schema = response_model.model_json_schema()
        cache_key = self.cache.make_key(
            {
                "cache_namespace": cache_namespace,
                "model": self.model,
                "messages": messages,
                "response_schema": response_schema,
            }
        )
        started = time.perf_counter()
        cached = self.cache.get(cache_key)
        if cached is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            zero_usage = TokenUsage()
            self.ledger.record(
                operation=operation,
                model=cached.model,
                request_id=None,
                cache_hit=True,
                usage=zero_usage,
                latency_ms=latency_ms,
                cost_usd=0.0,
                saved_cost_usd=cached.original_cost_usd,
                cache_type="exact",
            )
            return StructuredResult(
                data=cached.data,
                model=cached.model,
                request_id=None,
                usage=zero_usage,
                latency_ms=latency_ms,
                cache_hit=True,
                cost_usd=0.0,
                saved_cost_usd=cached.original_cost_usd,
                cache_type="exact",
            )

        semantic_vector: list[float] | None = None
        semantic_guard_key: str | None = None
        if semantic_cache is not None and self.semantic_encoder is not None:
            semantic_vector = [
                float(value) for value in self.semantic_encoder.encode(semantic_cache.text)
            ]
            semantic_guard_key = self.cache.make_key(
                {
                    "cache_namespace": cache_namespace,
                    "model": self.model,
                    "response_schema": response_schema,
                    "guard": semantic_cache.guard,
                }
            )
            match = self.cache.get_semantic(
                guard_key=semantic_guard_key,
                embedding_model=self.semantic_encoder.model_id,
                query_vector=semantic_vector,
                threshold=self.semantic_threshold,
            )
            if match is not None:
                latency_ms = (time.perf_counter() - started) * 1000
                zero_usage = TokenUsage()
                self.ledger.record(
                    operation=operation,
                    model=match.entry.model,
                    request_id=None,
                    cache_hit=True,
                    usage=zero_usage,
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                    saved_cost_usd=match.entry.original_cost_usd,
                    cache_type="semantic",
                    cache_similarity=match.similarity,
                )
                return StructuredResult(
                    data=match.entry.data,
                    model=match.entry.model,
                    request_id=None,
                    usage=zero_usage,
                    latency_ms=latency_ms,
                    cache_hit=True,
                    cost_usd=0.0,
                    saved_cost_usd=match.entry.original_cost_usd,
                    cache_type="semantic",
                    cache_similarity=match.similarity,
                )

        provider_result = self.provider.generate_structured(
            model=self.model,
            messages=messages,
            response_model=response_model,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        cost_usd = estimate_cost_usd(provider_result.model, provider_result.usage)
        self.cache.set(
            cache_key,
            data=provider_result.data,
            model=provider_result.model,
            usage=provider_result.usage,
            original_cost_usd=cost_usd,
        )
        if semantic_vector is not None and semantic_guard_key is not None:
            self.cache.set_semantic(
                guard_key=semantic_guard_key,
                source_cache_key=cache_key,
                embedding_model=self.semantic_encoder.model_id,
                vector=semantic_vector,
            )
        self.ledger.record(
            operation=operation,
            model=provider_result.model,
            request_id=provider_result.request_id,
            cache_hit=False,
            usage=provider_result.usage,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cache_type="none",
        )
        return StructuredResult(
            data=provider_result.data,
            model=provider_result.model,
            request_id=provider_result.request_id,
            usage=provider_result.usage,
            latency_ms=latency_ms,
            cache_hit=False,
            cost_usd=cost_usd,
            cache_type="none",
        )


def build_openai_client(settings: LLMSettings | None = None) -> CachedLLMClient:
    settings = settings or LLMSettings.from_env()
    semantic_encoder = (
        SentenceTransformerSemanticEncoder(
            settings.semantic_cache_model,
            device=settings.semantic_cache_device,
        )
        if settings.semantic_cache_enabled
        else None
    )
    return CachedLLMClient(
        provider=OpenAIResponsesProvider(settings),
        model=settings.model,
        cache=SQLiteLLMCache(settings.cache_path),
        ledger=SQLiteUsageLedger(settings.usage_path),
        semantic_encoder=semantic_encoder,
        semantic_threshold=settings.semantic_cache_threshold,
    )
