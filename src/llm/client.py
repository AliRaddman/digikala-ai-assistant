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
    ) -> None:
        self.provider = provider
        self.model = model
        self.cache = cache
        self.ledger = ledger

    def generate_structured(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]],
        response_model: type[SchemaT],
        cache_namespace: str,
    ) -> StructuredResult:
        cache_key = self.cache.make_key(
            {
                "cache_namespace": cache_namespace,
                "model": self.model,
                "messages": messages,
                "response_schema": response_model.model_json_schema(),
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
        self.ledger.record(
            operation=operation,
            model=provider_result.model,
            request_id=provider_result.request_id,
            cache_hit=False,
            usage=provider_result.usage,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        return StructuredResult(
            data=provider_result.data,
            model=provider_result.model,
            request_id=provider_result.request_id,
            usage=provider_result.usage,
            latency_ms=latency_ms,
            cache_hit=False,
            cost_usd=cost_usd,
        )


def build_openai_client(settings: LLMSettings | None = None) -> CachedLLMClient:
    settings = settings or LLMSettings.from_env()
    return CachedLLMClient(
        provider=OpenAIResponsesProvider(settings),
        model=settings.model,
        cache=SQLiteLLMCache(settings.cache_path),
        ledger=SQLiteUsageLedger(settings.usage_path),
    )
