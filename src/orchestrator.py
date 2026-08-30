"""Intent routing and chain orchestration for the shopping assistant.

Owner: Benyamin. This module deliberately depends on chain contracts, not on
the comment index. Missing teammate chains are registered later and currently
produce an explicit dependency status instead of crashing the whole system.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.chains.category_analytics import (
    DEFAULT_COMMENTS_PATH,
    DEFAULT_PRODUCTS_PATH,
    CategoryAnalyticsChain,
    load_products,
    resolve_category_from_query,
)
from src.chains.product_comparison import ProductComparisonChain
from src.chains.product_discovery import ProductDiscoveryChain
from src.chains.product_filters import RuleBasedFilterExtractor
from src.chains.product_qa import ProductQAChain
from src.data.normalize import normalize, to_search_text
from src.llm.client import CachedLLMClient, build_openai_client
from src.retrieval.base import Retriever, build_retriever

AssistantIntent = Literal[
    "product_discovery",
    "product_qa",
    "product_comparison",
    "category_analytics",
]
OrchestratorStatus = Literal[
    "success",
    "needs_input",
    "dependency_unavailable",
    "error",
]


class OrchestratorRequest(BaseModel):
    """One user turn plus any product context resolved by the UI/session."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    product_ids: list[str] = Field(default_factory=list)
    context_product_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = normalize(value)
        if not cleaned:
            raise ValueError("query cannot be empty")
        return cleaned

    @field_validator("product_ids", "context_product_ids", mode="before")
    @classmethod
    def stringify_ids(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise ValueError("product ids must be a list")
        cleaned = (str(item).strip() for item in value)
        return list(dict.fromkeys(item for item in cleaned if item))


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: AssistantIntent
    confidence: float = Field(ge=0, le=1)
    reason: str
    product_ids: list[str] = Field(default_factory=list)
    matched_signals: list[str] = Field(default_factory=list)


class HandlerResult(BaseModel):
    """Common output contract implemented by every feature-chain adapter."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class OrchestratorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: OrchestratorStatus
    route: RouteDecision
    answer: str
    citations: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    error: str | None = None


class IntentRouter(Protocol):
    def route(self, request: OrchestratorRequest) -> RouteDecision: ...


class ChainHandler(Protocol):
    def handle(self, request: OrchestratorRequest) -> HandlerResult: ...


_PRODUCT_TAG_RE = re.compile(r"\[product:([^\]\s]+)\]", re.IGNORECASE)
_EXPLICIT_PRODUCT_ID_RE = re.compile(
    r"(?:product(?:_id)?|شناسه(?:ی)?\s*(?:محصول|کالا)?|کالای)\s*[:#]?\s*(\d{3,})",
    re.IGNORECASE,
)

_ANALYTICS_SIGNALS = (
    "این دسته",
    "در این دسته",
    "سطح دسته",
    "پرتکرارترین شکایت",
    "پرتکرارترین مشکل",
    "چند برند اصلی",
    "مقایسه برندها",
    "کدام محصولات نظر زیادی",
    "درصد پیشنهاد خرید",
)
_COMPARISON_SIGNALS = (
    "مقایسه کن",
    "مقایسه‌شون",
    "مقایسه شون",
    "کدام بهتر",
    "کدوم بهتر",
    "فرق این دو",
    "در برابر",
    " versus ",
    " vs ",
)
_DISCOVERY_SIGNALS = (
    "معرفی کن",
    "پیشنهاد بده",
    "پیدا کن",
    "میخوام",
    "می خوام",
    "می خواهم",
    "دنبال ",
    "چند وسیله",
    "چند محصول",
    "چه محصولی",
)
_PRODUCT_QA_SIGNALS = (
    "این محصول",
    "این کالا",
    "نظرها درباره",
    "نظرات درباره",
    "ایرادهای پرتکرار",
    "نقاط قوت",
    "نقاط ضعف",
    "ارزش خرید",
    "تجربه خریداران",
    "کیفیتش چطوره",
)


def extract_product_ids(query: str) -> list[str]:
    """Extract citable/explicit product IDs without mistaking prices for IDs."""

    normalized = normalize(query)
    found = _PRODUCT_TAG_RE.findall(normalized)
    found.extend(_EXPLICIT_PRODUCT_ID_RE.findall(normalized))
    return list(dict.fromkeys(identifier.strip() for identifier in found))


class RuleBasedIntentRouter:
    """Free deterministic baseline with conservative, auditable precedence."""

    def route(self, request: OrchestratorRequest) -> RouteDecision:
        query = f" {to_search_text(request.query)} "
        explicit_product_ids = list(
            dict.fromkeys(
                [
                    *request.product_ids,
                    *extract_product_ids(request.query),
                ]
            )
        )
        product_ids = list(
            dict.fromkeys(
                [
                    *explicit_product_ids,
                    *request.context_product_ids,
                ]
            )
        )

        analytics = _matching_signals(query, _ANALYTICS_SIGNALS)
        comparison = _matching_signals(query, _COMPARISON_SIGNALS)
        discovery = _matching_signals(query, _DISCOVERY_SIGNALS)
        product_qa = _matching_signals(query, _PRODUCT_QA_SIGNALS)

        if analytics:
            return _decision(
                "category_analytics",
                0.98,
                "The request asks for an aggregate category or brand analysis.",
                product_ids,
                analytics,
            )
        if comparison or len(explicit_product_ids) >= 2:
            signals = comparison or ["multiple_product_ids"]
            return _decision(
                "product_comparison",
                0.97 if comparison else 0.90,
                "The request compares two or more products.",
                product_ids,
                signals,
            )
        # A shopping request can mention user satisfaction while still asking
        # for discovery. Explicit discovery language therefore wins over QA.
        if discovery:
            return _decision(
                "product_discovery",
                0.95,
                "The user asks the system to find or recommend products.",
                product_ids,
                discovery,
            )
        if product_qa or len(explicit_product_ids) == 1:
            signals = product_qa or ["single_product_id"]
            return _decision(
                "product_qa",
                0.92 if product_qa else 0.82,
                "The request asks about one identified product.",
                product_ids,
                signals,
            )
        return _decision(
            "product_discovery",
            0.60,
            "No strong specialized signal was found; product discovery is the safe default.",
            product_ids,
            ["default"],
        )


def _matching_signals(query: str, signals: tuple[str, ...]) -> list[str]:
    return [signal.strip() for signal in signals if signal in query]


def _decision(
    intent: AssistantIntent,
    confidence: float,
    reason: str,
    product_ids: list[str],
    signals: list[str],
) -> RouteDecision:
    return RouteDecision(
        intent=intent,
        confidence=confidence,
        reason=reason,
        product_ids=product_ids,
        matched_signals=signals,
    )


@dataclass(slots=True)
class ProductDiscoveryHandler:
    chain: ProductDiscoveryChain

    def handle(self, request: OrchestratorRequest) -> HandlerResult:
        result = self.chain.run(request.query, top_k=request.top_k)
        return HandlerResult(
            answer=result.render_fa(),
            citations=[item.citation() for item in result.products],
            payload=result.as_dict(),
        )


@lru_cache(maxsize=1)
def _default_llm_client() -> CachedLLMClient:
    """Built lazily, on first real use, so build_default_orchestrator() never
    requires an API key -- only an actual product_qa/category_analytics
    request does, and every other test keeps running fully offline."""
    return build_openai_client()


def _optional_llm_client() -> CachedLLMClient | None:
    """Same client, but None instead of an exception when no key is set.

    Used by category analytics and product comparison. Both compute their
    substance without a model -- analytics from pandas aggregation, comparison
    from retrieved product facts and review evidence -- and the model only
    narrates or infers on top, so with no key both still answer. Product QA
    deliberately does not degrade this way: there the model *is* the answer.
    """
    try:
        return _default_llm_client()
    except Exception:
        return None


@dataclass(slots=True)
class ProductQAHandler:
    retriever: Retriever
    max_evidence: int = 20
    client: CachedLLMClient | None = None
    """Overridable for tests; production code leaves this None and gets the
    lazily-built default client, only on the first real request."""

    def handle(self, request: OrchestratorRequest) -> HandlerResult:
        chain = ProductQAChain(
            retriever=self.retriever,
            client=self.client or _default_llm_client(),
            max_evidence=self.max_evidence,
        )
        result = chain.run(request.query, request.product_ids[0])
        return HandlerResult(
            answer=result.render_fa(),
            citations=[item.citation() for item in result.evidence],
            payload=result.as_dict(),
        )


@dataclass(slots=True)
class ProductComparisonHandler:
    """Adapter for Fatemeh's ProductComparisonChain.

    Two retrievers rather than one: the chain resolves each selected product
    through the product index and then pulls product-scoped review evidence
    through the comment index. Fatemeh's notebook left the review side empty
    because the comment index did not exist yet; it does now, so this wires
    both and the evidence section of the answer is populated.

    The LLM client is optional on purpose. ProductComparisonChain renders
    facts and review evidence with no model at all and only skips the
    inference section, so the comparison path stays usable without an API key
    -- same degradation as category analytics, and the reason this uses
    _optional_llm_client rather than _default_llm_client.
    """

    product_retriever: Retriever
    comment_retriever: Retriever
    comment_top_k: int = 5
    client: CachedLLMClient | None = None
    """Overridable for tests; see ProductQAHandler.client."""

    def handle(self, request: OrchestratorRequest) -> HandlerResult:
        chain = ProductComparisonChain(
            product_retriever=self.product_retriever,
            comment_retriever=self.comment_retriever,
            llm_client=self.client or _optional_llm_client(),
            comment_top_k=self.comment_top_k,
        )
        result = chain.run(request.query, product_ids=request.product_ids)
        return HandlerResult(
            answer=result.render_fa(),
            citations=result.citations(),
            payload=result.as_dict(),
        )


_CATEGORY_NOT_RESOLVED_FA = (
    "متوجه نشدم منظورتان کدام دسته‌ی محصول است؛ لطفاً نام دسته را در پرسش "
    "بیاورید (مثلاً «اسباب‌بازی» یا «لباس زنانه»)."
)


@dataclass(slots=True)
class CategoryAnalyticsHandler:
    """Resolves the category itself rather than relying on RouteDecision:
    matching free text against the real category vocabulary needs
    products_clean_v1.parquet, and RuleBasedIntentRouter is meant to stay a
    zero-file-I/O baseline that every routing test can call without data on
    disk. Doing it here only costs a disk read on an actual category_analytics
    request, exactly like the retriever/LLM client are only touched then too.
    """

    products_path: Path = DEFAULT_PRODUCTS_PATH
    comments_path: Path = DEFAULT_COMMENTS_PATH
    client: CachedLLMClient | None = None
    """Overridable for tests; see ProductQAHandler.client."""

    def handle(self, request: OrchestratorRequest) -> HandlerResult:
        products = load_products(str(self.products_path))
        scope = resolve_category_from_query(request.query, products)
        if scope is None:
            return HandlerResult(answer=_CATEGORY_NOT_RESOLVED_FA)
        chain = CategoryAnalyticsChain(
            products_path=self.products_path,
            comments_path=self.comments_path,
            client=self.client or _optional_llm_client(),
        )
        report = chain.run(scope)
        return HandlerResult(
            answer=report.render_fa(),
            citations=[],
            payload=report.as_dict(),
        )


_DEPENDENCIES: dict[AssistantIntent, list[str]] = {
    "product_discovery": ["product_discovery_chain"],
    "product_qa": ["product_qa_chain", "comment_retriever"],
    "product_comparison": ["product_comparison_chain"],
    "category_analytics": ["category_analytics_chain"],
}


class ShoppingAssistantOrchestrator:
    """Route one request and invoke only the selected registered chain."""

    def __init__(
        self,
        *,
        router: IntentRouter,
        handlers: Mapping[AssistantIntent, ChainHandler],
    ) -> None:
        self.router = router
        self.handlers = dict(handlers)

    def run(
        self,
        query: str,
        *,
        product_ids: list[str] | None = None,
        context_product_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> OrchestratorResult:
        request = OrchestratorRequest(
            query=query,
            product_ids=product_ids or [],
            context_product_ids=context_product_ids or [],
            top_k=top_k,
        )
        decision = self.router.route(request)
        missing_input = _missing_input(decision)
        if missing_input:
            return OrchestratorResult(
                status="needs_input",
                route=decision,
                answer=_missing_input_message(decision.intent),
                missing_requirements=missing_input,
            )

        handler = self.handlers.get(decision.intent)
        if handler is None:
            return OrchestratorResult(
                status="dependency_unavailable",
                route=decision,
                answer=_dependency_message(decision.intent),
                missing_requirements=_DEPENDENCIES[decision.intent],
            )

        routed_request = request.model_copy(
            update={"product_ids": decision.product_ids}
        )
        try:
            handled = handler.handle(routed_request)
        except Exception as exc:
            return OrchestratorResult(
                status="error",
                route=decision,
                answer="در اجرای بخش انتخاب‌شده خطایی رخ داد.",
                error=f"{type(exc).__name__}: {exc}",
            )
        return OrchestratorResult(
            status="success",
            route=decision,
            answer=handled.answer,
            citations=handled.citations,
            payload=handled.payload,
        )


def _missing_input(decision: RouteDecision) -> list[str]:
    if decision.intent == "product_qa" and not decision.product_ids:
        return ["one_product_id"]
    if decision.intent == "product_comparison" and len(decision.product_ids) < 2:
        return ["at_least_two_product_ids"]
    return []


def _missing_input_message(intent: AssistantIntent) -> str:
    if intent == "product_qa":
        return "برای پاسخ بر اساس نظرات، لطفاً شناسه محصول را مشخص کنید."
    return "برای مقایسه، لطفاً شناسه حداقل دو محصول را مشخص کنید."


def _dependency_message(intent: AssistantIntent) -> str:
    messages = {
        "product_discovery": "بخش جست‌وجوی محصول هنوز متصل نشده است.",
        "product_qa": "مسیر پرسش‌وپاسخ محصول شناسایی شد، اما Comment Retriever هنوز متصل نیست.",
        "product_comparison": "مسیر مقایسه شناسایی شد، اما Chain مقایسه هنوز متصل نیست.",
        "category_analytics": "مسیر تحلیل دسته شناسایی شد، اما Chain تحلیلی هنوز متصل نیست.",
    }
    return messages[intent]


def build_default_orchestrator(
    *,
    retriever_mode: str = "mock",
) -> ShoppingAssistantOrchestrator:
    """Build today's runnable system; teammate handlers can be added later."""

    discovery_retriever = build_retriever("product", mode=retriever_mode)
    discovery = ProductDiscoveryHandler(
        ProductDiscoveryChain(
            retriever=discovery_retriever,
            extractor=RuleBasedFilterExtractor(),
        )
    )
    # Built once and shared by the QA and comparison handlers rather than once
    # per handler: in real mode each build loads the comment index and its
    # 10.6 GB memmap, and two of those would double the startup cost for no
    # benefit -- retrievers hold no per-request state.
    product_retriever = discovery_retriever
    comment_retriever = build_retriever("comment", mode=retriever_mode)

    product_qa = ProductQAHandler(retriever=comment_retriever)
    product_comparison = ProductComparisonHandler(
        product_retriever=product_retriever,
        comment_retriever=comment_retriever,
    )
    category_analytics = CategoryAnalyticsHandler()

    return ShoppingAssistantOrchestrator(
        router=RuleBasedIntentRouter(),
        handlers={
            "product_discovery": discovery,
            "product_qa": product_qa,
            "product_comparison": product_comparison,
            "category_analytics": category_analytics,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a Persian shopping request.")
    parser.add_argument("query")
    parser.add_argument("--product-id", action="append", default=[])
    parser.add_argument("--context-product-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever-mode", default="mock")
    args = parser.parse_args()

    result = build_default_orchestrator(
        retriever_mode=args.retriever_mode
    ).run(
        args.query,
        product_ids=args.product_id,
        context_product_ids=args.context_product_id,
        top_k=args.top_k,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
