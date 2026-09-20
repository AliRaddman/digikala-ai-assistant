"""Grounded product-comparison chain.

Owner: Fatemeh.

The chain follows the project's shared contracts:
- product and comment access only through src.retrieval.base.Retriever
- optional structured inference through Benyamin's CachedLLMClient
- facts / evidence / inference remain separate
- no API key is read or stored here

The orchestrator passes at least two product_ids. The chain retrieves the
selected products, gathers product-scoped review evidence, and optionally asks
the shared cached LLM client for a grounded comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.data.normalize import normalize
from src.llm.client import CachedLLMClient
from src.llm.semantic_cache import SemanticCacheRequest
from src.retrieval.base import Evidence, RetrievalFilters, Retriever


PROMPT_VERSION = "product-comparison-v1"

SYSTEM_PROMPT = """You are a grounded Persian shopping-comparison assistant.

You receive structured product facts and retrieved user-review evidence.

Rules:
- Use ONLY the supplied facts and review evidence.
- Do not use outside product knowledge.
- Keep direct facts, review evidence, and inference conceptually separate.
- Every factual/review-derived comparison claim must cite one or more supplied
  citation tags such as [product:123] or [comment:456].
- Never invent a citation.
- If the evidence is insufficient for a criterion, say so explicitly.
- Do not treat retrieval score as product quality.
- Do not claim that one product is globally better unless the supplied evidence
  supports that conclusion.
- Answer in Persian.
"""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_citation(value: Any) -> str | None:
    """`comment:456` and `[comment:456]` both become `[comment:456]`.

    Fixed by Ali, 2026-08-30, with Fatemeh unavailable. _sanitize_inference
    compared the model's citations against Evidence.citation() by exact string
    match. SYSTEM_PROMPT above asks for bracketed tags, but the identically
    worded instruction in src/eval/grounding.py produced bare `product:123`
    from the same model on the first live run, and there the mismatch threw
    away 34 paid-for judgments (failure 9 in docs/FAILURES.md).

    Here it would not have raised: the filter drops what it does not
    recognise, so a comparison would have rendered with every citation
    silently removed and still looked healthy -- the worst shape for a chain
    whose whole value is being grounded in cited evidence.

    Only the brackets are forgiven. The id itself is still matched verbatim
    against the supplied evidence, so an invented citation is dropped exactly
    as before, and the value kept is the canonical bracketed form so that
    render_fa and grounding.audit_citations see one consistent spelling.
    """
    if not isinstance(value, str):
        return None
    bare = value.strip().removeprefix("[").removesuffix("]").strip()
    return f"[{bare}]" if bare else None


@dataclass(frozen=True, slots=True)
class ProductFacts:
    product_id: str
    title: str
    brand: str | None
    price_rial: float | None
    rate: float | None
    rate_count: int | None
    cat1: str | None
    sub_cat: str | None
    is_fake: bool | None
    retrieval_score: float
    citation: str

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "ProductFacts":
        return cls(
            product_id=str(evidence.product_id or evidence.id),
            title=str(evidence.title or evidence.text),
            brand=evidence.meta.get("brand"),
            price_rial=_optional_float(evidence.meta.get("price")),
            rate=_optional_float(evidence.meta.get("rate")),
            rate_count=_optional_int(evidence.meta.get("rate_count")),
            cat1=evidence.meta.get("cat1"),
            sub_cat=evidence.meta.get("sub_cat"),
            is_fake=(
                None
                if evidence.meta.get("is_fake") is None
                else bool(evidence.meta.get("is_fake"))
            ),
            retrieval_score=float(evidence.score),
            citation=evidence.citation(),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProductReviewEvidence:
    product_id: str
    query: str
    comments: list[Evidence]

    @property
    def available(self) -> bool:
        return bool(self.comments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "query": self.query,
            "available": self.available,
            "items": [comment.as_dict() for comment in self.comments],
        }


class ProductAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class CriterionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str
    better_product_id: str | None = None
    explanation: str
    citations: list[str] = Field(default_factory=list)


class ComparisonInference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_assessments: list[ProductAssessment] = Field(default_factory=list)
    criteria: list[CriterionDecision] = Field(default_factory=list)
    overall_winner_product_id: str | None = None
    overall_recommendation: str
    caveats: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ProductComparisonResult:
    query: str
    requested_product_ids: list[str]
    facts: list[ProductFacts]
    evidence: list[ProductReviewEvidence]
    inference: ComparisonInference | None
    missing_products: list[str]
    warnings: list[str]

    def citations(self) -> list[str]:
        citations: list[str] = []

        for fact in self.facts:
            citations.append(fact.citation)

        for group in self.evidence:
            citations.extend(comment.citation() for comment in group.comments)

        return list(dict.fromkeys(citations))

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "requested_product_ids": self.requested_product_ids,
            "facts": [fact.as_dict() for fact in self.facts],
            "evidence": {
                "available": any(group.available for group in self.evidence),
                "items": [group.as_dict() for group in self.evidence],
            },
            "inference": (
                None
                if self.inference is None
                else self.inference.model_dump(mode="json")
            ),
            "missing_products": self.missing_products,
            "warnings": self.warnings,
        }

    def render_fa(self) -> str:
        lines: list[str] = []

        if self.facts:
            lines.append("مقایسه محصولات:")
            for fact in self.facts:
                details: list[str] = []
                if fact.brand:
                    details.append(f"برند: {fact.brand}")
                if fact.price_rial is not None:
                    details.append(f"قیمت: {fact.price_rial / 10:,.0f} تومان")
                if fact.rate is not None:
                    details.append(f"امتیاز: {fact.rate:g}")
                if fact.rate_count is not None:
                    details.append(f"تعداد امتیاز: {fact.rate_count:,}")

                suffix = " | ".join(details)
                if suffix:
                    lines.append(f"- {fact.title} {fact.citation} — {suffix}")
                else:
                    lines.append(f"- {fact.title} {fact.citation}")

        evidence_by_pid = {group.product_id: group for group in self.evidence}
        if self.facts:
            lines.append("")
            lines.append("شواهد نظرات:")
            for fact in self.facts:
                group = evidence_by_pid.get(fact.product_id)
                count = len(group.comments) if group else 0
                lines.append(
                    f"- {fact.title}: {count} نظر مرتبط بازیابی شد."
                )

        if self.inference is not None:
            lines.append("")
            lines.append("جمع‌بندی استنباطی:")
            lines.append(self.inference.overall_recommendation)

            if self.inference.product_assessments:
                assessment_by_pid = {
                    item.product_id: item
                    for item in self.inference.product_assessments
                }
                for fact in self.facts:
                    assessment = assessment_by_pid.get(fact.product_id)
                    if assessment is None:
                        continue

                    if assessment.strengths:
                        lines.append(
                            f"- نقاط قوت {fact.title}: "
                            + "؛ ".join(assessment.strengths)
                        )
                    if assessment.weaknesses:
                        lines.append(
                            f"- نقاط ضعف {fact.title}: "
                            + "؛ ".join(assessment.weaknesses)
                        )

            for criterion in self.inference.criteria:
                cite_text = " ".join(criterion.citations)
                winner = (
                    f"محصول {criterion.better_product_id}"
                    if criterion.better_product_id
                    else "برنده مشخص نیست"
                )
                lines.append(
                    f"- {criterion.criterion}: {winner} — "
                    f"{criterion.explanation} {cite_text}".strip()
                )

            if self.inference.caveats:
                lines.append("")
                lines.append("محدودیت‌ها:")
                lines.extend(f"- {item}" for item in self.inference.caveats)
        else:
            lines.append("")
            lines.append(
                "استنباط LLM غیرفعال است؛ خروجی بالا فقط facts و review evidence است."
            )

        if self.missing_products:
            lines.append("")
            lines.append(
                "شناسه‌های محصول پیدا نشد: "
                + ", ".join(self.missing_products)
            )

        if self.warnings:
            lines.append("")
            lines.append("هشدارها:")
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines).strip()


class ProductComparisonChain:
    """Compare selected products using shared product/comment retrievers."""

    def __init__(
        self,
        *,
        product_retriever: Retriever,
        comment_retriever: Retriever,
        llm_client: CachedLLMClient | None = None,
        comment_top_k: int = 5,
    ) -> None:
        if comment_top_k < 1:
            raise ValueError("comment_top_k must be at least 1")

        self.product_retriever = product_retriever
        self.comment_retriever = comment_retriever
        self.llm_client = llm_client
        self.comment_top_k = comment_top_k

    def run(
        self,
        query: str,
        *,
        product_ids: list[str],
        comment_top_k: int | None = None,
    ) -> ProductComparisonResult:
        clean_query = normalize(query)
        if not clean_query:
            raise ValueError("query cannot be empty")

        clean_product_ids = list(
            dict.fromkeys(
                cleaned
                for product_id in product_ids
                if product_id is not None
                for cleaned in [str(product_id).strip()]
                if cleaned
            )
        )
        if len(clean_product_ids) < 2:
            raise ValueError("At least two product_ids are required")

        top_k_comments = (
            self.comment_top_k
            if comment_top_k is None
            else comment_top_k
        )
        if top_k_comments < 1:
            raise ValueError("comment_top_k must be at least 1")

        facts, missing_products, product_warnings = self._retrieve_facts(
            clean_query,
            clean_product_ids,
        )

        review_evidence: list[ProductReviewEvidence] = []
        warnings = list(product_warnings)

        for fact in facts:
            evidence_query = self._build_comment_query(clean_query, fact)
            comments = self.comment_retriever.retrieve(
                evidence_query,
                top_k=top_k_comments,
                filters=RetrievalFilters(product_ids=[fact.product_id]),
            )

            # Defensive check: a scoped retriever must not leak comments from
            # another product even if a concrete implementation is buggy.
            scoped_comments: list[Evidence] = []
            seen_comment_ids: set[str] = set()

            for comment in comments:
                if comment.kind != "comment":
                    continue
                if str(comment.product_id) != fact.product_id:
                    continue
                if comment.id in seen_comment_ids:
                    continue

                seen_comment_ids.add(comment.id)
                scoped_comments.append(comment)

            comments = scoped_comments

            if not comments:
                warnings.append(
                    f"No review evidence was retrieved for product {fact.product_id}."
                )

            review_evidence.append(
                ProductReviewEvidence(
                    product_id=fact.product_id,
                    query=evidence_query,
                    comments=comments,
                )
            )

        inference: ComparisonInference | None = None
        if self.llm_client is not None and len(facts) >= 2:
            inference = self._generate_inference(
                clean_query,
                facts,
                review_evidence,
            )

        return ProductComparisonResult(
            query=clean_query,
            requested_product_ids=clean_product_ids,
            facts=facts,
            evidence=review_evidence,
            inference=inference,
            missing_products=missing_products,
            warnings=warnings,
        )

    def _retrieve_facts(
        self,
        query: str,
        product_ids: list[str],
    ) -> tuple[list[ProductFacts], list[str], list[str]]:
        """Retrieve each selected product using the shared hard-ID filter."""

        found: dict[str, ProductFacts] = {}
        warnings: list[str] = []

        # Query each ID separately so one selected product cannot crowd another
        # out of the retriever's top-k result.
        for product_id in product_ids:
            results = self.product_retriever.retrieve(
                query,
                top_k=5,
                filters=RetrievalFilters(product_ids=[product_id]),
            )

            exact = next(
                (
                    item
                    for item in results
                    if item.kind == "product"
                    and str(item.product_id or item.id) == product_id
                ),
                None,
            )

            if exact is None:
                # One conservative fallback still stays behind the shared
                # Retriever contract. It helps mock/lexical backends whose
                # first query contains little product-specific text.
                fallback_query = f"{query} {product_id}"
                fallback_results = self.product_retriever.retrieve(
                    fallback_query,
                    top_k=10,
                    filters=RetrievalFilters(product_ids=[product_id]),
                )
                exact = next(
                    (
                        item
                        for item in fallback_results
                        if item.kind == "product"
                    and str(item.product_id or item.id) == product_id
                    ),
                    None,
                )

            if exact is None:
                continue

            fact = ProductFacts.from_evidence(exact)
            if fact.product_id in found:
                warnings.append(
                    f"Duplicate product resolution for product_id={fact.product_id}."
                )
                continue

            found[fact.product_id] = fact

        facts = [
            found[product_id]
            for product_id in product_ids
            if product_id in found
        ]
        missing = [
            product_id
            for product_id in product_ids
            if product_id not in found
        ]

        return facts, missing, warnings

    @staticmethod
    def _build_comment_query(query: str, fact: ProductFacts) -> str:
        return normalize(
            f"{query} | {fact.title} | "
            "کیفیت نقاط قوت نقاط ضعف ایراد تجربه خرید ارزش خرید"
        )

    def _generate_inference(
        self,
        query: str,
        facts: list[ProductFacts],
        evidence: list[ProductReviewEvidence],
    ) -> ComparisonInference:
        prompt_payload = {
            "user_query": query,
            "facts": [fact.as_dict() for fact in facts],
            "review_evidence": [
                {
                    "product_id": group.product_id,
                    "comments": [
                        {
                            "citation": comment.citation(),
                            "text": comment.text[:500],
                            "rate": comment.meta.get("rate"),
                            "recommendation_status": comment.meta.get(
                                "recommendation_status"
                            ),
                            "is_buyer": comment.meta.get("is_buyer"),
                            "likes": comment.meta.get("likes"),
                            "advantages": comment.meta.get("advantages"),
                            "disadvantages": comment.meta.get("disadvantages"),
                        }
                        for comment in group.comments
                    ],
                }
                for group in evidence
            ],
        }

        result = self.llm_client.generate_structured(
            operation="product_comparison",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        prompt_payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            response_model=ComparisonInference,
            cache_namespace=PROMPT_VERSION,
            semantic_cache=SemanticCacheRequest(
                text=query,
                guard={
                    "system_prompt": SYSTEM_PROMPT,
                    "facts": prompt_payload["facts"],
                    "review_evidence": prompt_payload["review_evidence"],
                },
            ),
        )

        raw = dict(result.data)
        allowed_citations = {
            fact.citation for fact in facts
        } | {
            comment.citation()
            for group in evidence
            for comment in group.comments
        }
        allowed_product_ids = {fact.product_id for fact in facts}

        cleaned = self._sanitize_inference(
            raw,
            allowed_citations=allowed_citations,
            allowed_product_ids=allowed_product_ids,
        )
        return ComparisonInference.model_validate(cleaned)

    @classmethod
    def _sanitize_inference(
        cls,
        value: Any,
        *,
        allowed_citations: set[str],
        allowed_product_ids: set[str],
    ) -> Any:
        """Remove invented citations/product IDs from an otherwise valid result."""

        if isinstance(value, list):
            return [
                cls._sanitize_inference(
                    item,
                    allowed_citations=allowed_citations,
                    allowed_product_ids=allowed_product_ids,
                )
                for item in value
            ]

        if not isinstance(value, dict):
            return value

        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "product_assessments" and isinstance(item, list):
                cleaned[key] = [
                    cls._sanitize_inference(
                        assessment,
                        allowed_citations=allowed_citations,
                        allowed_product_ids=allowed_product_ids,
                    )
                    for assessment in item
                    if isinstance(assessment, dict)
                    and assessment.get("product_id") in allowed_product_ids
                ]
                continue

            if key == "citations" and isinstance(item, list):
                cleaned[key] = [
                    canonical
                    for citation in item
                    for canonical in [_canonical_citation(citation)]
                    if canonical in allowed_citations
                ]
                continue

            if key in {"better_product_id", "overall_winner_product_id"}:
                cleaned[key] = (
                    item if item in allowed_product_ids else None
                )
                continue

            if key == "product_id":
                cleaned[key] = (
                    item if item in allowed_product_ids else ""
                )
                continue

            cleaned[key] = cls._sanitize_inference(
                item,
                allowed_citations=allowed_citations,
                allowed_product_ids=allowed_product_ids,
            )

        return cleaned
