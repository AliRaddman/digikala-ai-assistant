"""Reproducible evaluation for the grounded product-comparison chain.

The historical comparison run on Fatemeh's branch checked retrieval structure
only: the LLM client was disabled, so no comparison inference existed.  This
module keeps that useful structural check, but reports it separately from
citation integrity and optional LLM-as-a-Judge scores.  A structural pass must
never be presented as evidence that the generated comparison is good.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.chains.product_comparison import (
    ProductComparisonChain,
    ProductComparisonResult,
    ProductFacts,
)
from src.eval.grounding import CitationAudit, GroundingJudge, audit_citations
from src.llm.usage import SQLiteUsageLedger
from src.retrieval.base import Evidence


class ComparisonEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    category: str | None = None
    query: str = Field(min_length=1)
    product_ids: list[str]

    @field_validator("product_ids", mode="before")
    @classmethod
    def stringify_product_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("product_ids must be a list")
        cleaned = [str(product_id).strip() for product_id in value]
        return [product_id for product_id in cleaned if product_id]

    @model_validator(mode="after")
    def require_distinct_products(self) -> "ComparisonEvalCase":
        if len(self.product_ids) < 2:
            raise ValueError("at least two product_ids are required")
        if len(set(self.product_ids)) != len(self.product_ids):
            raise ValueError("product_ids must be distinct")
        return self


class ComparisonStructuralAudit(BaseModel):
    """Retrieval/pipeline checks; this does not score generated prose quality."""

    model_config = ConfigDict(extra="forbid")

    facts_complete: bool
    no_missing_products: bool
    review_groups_complete: bool
    all_products_have_evidence: bool
    product_scope_consistent: bool
    structural_pass: bool
    facts_count: int
    evidence_counts: dict[str, int]
    missing_product_ids: list[str]
    unexpected_product_ids: list[str]


class ComparisonEvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    pair_id: str
    category: str | None
    query: str
    product_ids: list[str]
    first_case: bool
    chain_latency_ms: float
    end_to_end_latency_ms: float
    structural_audit: ComparisonStructuralAudit | None = None
    inference_available: bool = False
    rendered_citation_audit: CitationAudit | None = None
    inference_citation_audit: CitationAudit | None = None
    grounding_judgment: dict[str, Any] | None = None
    grounding_skip_reason: str | None = None
    judge_error: str | None = None
    answer_fa: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class ProductComparisonEvaluator:
    """Evaluate structure for every case and semantics only when available."""

    def __init__(
        self,
        *,
        chain: ProductComparisonChain,
        ledger: SQLiteUsageLedger | None = None,
        grounding_judge: GroundingJudge | None = None,
    ) -> None:
        self.chain = chain
        self.ledger = ledger
        self.grounding_judge = grounding_judge

    def evaluate(self, cases: list[ComparisonEvalCase]) -> dict[str, Any]:
        if not cases:
            raise ValueError("comparison evaluation set cannot be empty")

        ledger_checkpoint = self.ledger.checkpoint() if self.ledger else 0
        run_started = time.perf_counter()
        items = [
            self._evaluate_one(case, first_case=index == 0)
            for index, case in enumerate(cases)
        ]
        wall_time_ms = (time.perf_counter() - run_started) * 1000

        return {
            "schema_version": "product-comparison-eval-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration": {
                "product_retriever": type(self.chain.product_retriever).__name__,
                "comment_retriever": type(self.chain.comment_retriever).__name__,
                "comment_top_k": self.chain.comment_top_k,
                "llm_inference_enabled": self.chain.llm_client is not None,
                "grounding_judge": (
                    type(self.grounding_judge).__name__
                    if self.grounding_judge
                    else None
                ),
            },
            "summary": _summarize(items, wall_time_ms),
            "llm_usage": (
                self.ledger.summary(after_id=ledger_checkpoint)
                if self.ledger
                else None
            ),
            "items": [item.model_dump(mode="json") for item in items],
        }

    def _evaluate_one(
        self,
        case: ComparisonEvalCase,
        *,
        first_case: bool,
    ) -> ComparisonEvalItem:
        item_started = time.perf_counter()
        chain_started = time.perf_counter()
        try:
            result = self.chain.run(
                case.query,
                product_ids=case.product_ids,
            )
        except Exception as exc:  # one bad pair must not discard the full run
            chain_latency_ms = (time.perf_counter() - chain_started) * 1000
            return ComparisonEvalItem(
                case_id=case.case_id,
                pair_id=case.pair_id,
                category=case.category,
                query=case.query,
                product_ids=case.product_ids,
                first_case=first_case,
                chain_latency_ms=chain_latency_ms,
                end_to_end_latency_ms=(time.perf_counter() - item_started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        chain_latency_ms = (time.perf_counter() - chain_started) * 1000
        answer = result.render_fa()
        evidence = _result_evidence(result)
        structural_audit = _audit_structure(result, case.product_ids)
        rendered_audit = audit_citations(answer, evidence)
        inference_audit = (
            audit_citations(
                json.dumps(
                    result.inference.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                evidence,
            )
            if result.inference is not None
            else None
        )

        grounding_judgment: dict[str, Any] | None = None
        grounding_skip_reason: str | None = None
        judge_error: str | None = None
        if self.grounding_judge is None:
            grounding_skip_reason = "grounding judge disabled"
        elif result.inference is None:
            grounding_skip_reason = "LLM inference unavailable"
        else:
            try:
                grounding_judgment = self.grounding_judge.evaluate(
                    question=case.query,
                    answer=answer,
                    evidence=evidence,
                ).as_dict()
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: {exc}"

        return ComparisonEvalItem(
            case_id=case.case_id,
            pair_id=case.pair_id,
            category=case.category,
            query=case.query,
            product_ids=case.product_ids,
            first_case=first_case,
            chain_latency_ms=chain_latency_ms,
            end_to_end_latency_ms=(time.perf_counter() - item_started) * 1000,
            structural_audit=structural_audit,
            inference_available=result.inference is not None,
            rendered_citation_audit=rendered_audit,
            inference_citation_audit=inference_audit,
            grounding_judgment=grounding_judgment,
            grounding_skip_reason=grounding_skip_reason,
            judge_error=judge_error,
            answer_fa=answer,
            result=result.as_dict(),
        )


def load_comparison_cases(path: Path) -> list[ComparisonEvalCase]:
    cases: list[ComparisonEvalCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = ComparisonEvalCase.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid JSONL record at line {line_number}: {exc}"
                ) from exc
            if case.case_id in seen_ids:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"no comparison cases found in {path}")
    return cases


def _audit_structure(
    result: ProductComparisonResult,
    expected_product_ids: list[str],
) -> ComparisonStructuralAudit:
    expected = [str(product_id) for product_id in expected_product_ids]
    fact_ids = [fact.product_id for fact in result.facts]
    group_ids = [group.product_id for group in result.evidence]
    expected_set = set(expected)

    evidence_counts = {product_id: 0 for product_id in expected}
    for group in result.evidence:
        evidence_counts[group.product_id] = len(group.comments)

    unexpected = sorted((set(fact_ids) | set(group_ids)) - expected_set)
    facts_complete = fact_ids == expected
    no_missing = not result.missing_products
    review_groups_complete = group_ids == expected
    all_products_have_evidence = all(
        evidence_counts.get(product_id, 0) > 0 for product_id in expected
    )
    product_scope_consistent = not unexpected and all(
        str(comment.product_id) == group.product_id
        for group in result.evidence
        for comment in group.comments
    )
    structural_pass = all(
        (
            facts_complete,
            no_missing,
            review_groups_complete,
            all_products_have_evidence,
            product_scope_consistent,
        )
    )
    return ComparisonStructuralAudit(
        facts_complete=facts_complete,
        no_missing_products=no_missing,
        review_groups_complete=review_groups_complete,
        all_products_have_evidence=all_products_have_evidence,
        product_scope_consistent=product_scope_consistent,
        structural_pass=structural_pass,
        facts_count=len(result.facts),
        evidence_counts=evidence_counts,
        missing_product_ids=result.missing_products,
        unexpected_product_ids=unexpected,
    )


def _result_evidence(result: ProductComparisonResult) -> list[Evidence]:
    product_evidence = [_fact_as_evidence(fact) for fact in result.facts]
    comment_evidence = [
        comment
        for group in result.evidence
        for comment in group.comments
    ]
    return product_evidence + comment_evidence


def _fact_as_evidence(fact: ProductFacts) -> Evidence:
    return Evidence(
        id=fact.product_id,
        kind="product",
        text=fact.title,
        score=fact.retrieval_score,
        product_id=fact.product_id,
        title=fact.title,
        meta={
            "brand": fact.brand,
            "price": fact.price_rial,
            "rate": fact.rate,
            "rate_count": fact.rate_count,
            "cat1": fact.cat1,
            "sub_cat": fact.sub_cat,
            "is_fake": fact.is_fake,
        },
    )


def _summarize(
    items: list[ComparisonEvalItem],
    wall_time_ms: float,
) -> dict[str, Any]:
    successful = [item for item in items if item.error is None]
    structural = [
        item.structural_audit
        for item in successful
        if item.structural_audit is not None
    ]
    inferred = [item for item in successful if item.inference_available]
    inference_audits = [
        item.inference_citation_audit
        for item in inferred
        if item.inference_citation_audit is not None
    ]
    inference_with_citations = [audit for audit in inference_audits if audit.found]
    judged = [item for item in successful if item.grounding_judgment is not None]

    judge_errors = sum(item.judge_error is not None for item in successful)
    quality_status = (
        "not_measured"
        if not inferred
        else "judge_failed"
        if judge_errors and not judged
        else "citation_only"
        if not judged
        else "judge_measured"
    )
    quality_note = {
        "not_measured": (
            "No LLM inference was generated. Structural passes measure retrieval "
            "and pipeline completeness only, not comparison quality."
        ),
        "citation_only": (
            "Inference exists, but no grounding judge ran. Citation integrity "
            "checks ID existence only, not whether evidence entails each claim."
        ),
        "judge_failed": (
            "Inference exists and a grounding judge was requested, but every "
            "judge call failed. Do not infer semantic quality from citation IDs."
        ),
        "judge_measured": (
            "LLM inference and grounding judgments are present. Human validation "
            "is still required before treating judge scores as ground truth."
        ),
    }[quality_status]

    rendered_audits = [
        item.rendered_citation_audit
        for item in successful
        if item.rendered_citation_audit is not None
        and item.rendered_citation_audit.integrity_score is not None
    ]
    remaining = items[1:]
    return {
        "cases": len(items),
        "errored": len(items) - len(successful),
        "structural_passed": sum(audit.structural_pass for audit in structural),
        "structural_pass_rate": _ratio(
            sum(audit.structural_pass for audit in structural),
            len(structural),
        ),
        "all_products_have_evidence_rate": _ratio(
            sum(audit.all_products_have_evidence for audit in structural),
            len(structural),
        ),
        "inference_available": len(inferred),
        "inference_coverage": _ratio(len(inferred), len(successful)),
        "citation_integrity": {
            "definition": "citation ID existence only; not claim entailment",
            "rendered_cases_audited": len(rendered_audits),
            "rendered_mean_integrity": _mean_optional(
                audit.integrity_score for audit in rendered_audits
            ),
            "inference_cases_audited": len(inference_audits),
            "inference_cases_with_citations": len(inference_with_citations),
            "inference_mean_integrity": _mean_optional(
                audit.integrity_score for audit in inference_with_citations
            ),
            "inference_invalid_ids": sum(
                len(audit.invalid) for audit in inference_audits
            ),
        },
        "grounding_judged": len(judged),
        "judge_errors": judge_errors,
        "judge_scores": {
            "mean_grounding": _mean_optional(
                item.grounding_judgment["judgment"]["grounding_score"]
                for item in judged
                if item.grounding_judgment is not None
            ),
            "mean_relevance": _mean_optional(
                item.grounding_judgment["judgment"]["relevance_score"]
                for item in judged
                if item.grounding_judgment is not None
            ),
            "verdict_counts": {
                verdict: sum(
                    item.grounding_judgment["judgment"]["verdict"] == verdict
                    for item in judged
                    if item.grounding_judgment is not None
                )
                for verdict in (
                    "grounded",
                    "partially_grounded",
                    "ungrounded",
                )
            },
        },
        "quality_assessment": {
            "status": quality_status,
            "note": quality_note,
        },
        "latency_ms": {
            "wall_time": wall_time_ms,
            "first_case": items[0].end_to_end_latency_ms,
            "remaining_cases": _distribution(
                [item.end_to_end_latency_ms for item in remaining]
            ),
            "all_chain_calls": _distribution(
                [item.chain_latency_ms for item in items]
            ),
        },
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean_optional(values: Any) -> float | None:
    materialized = [value for value in values if value is not None]
    return fmean(materialized) if materialized else None


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values) if values else 0.0,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values, default=0.0),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
