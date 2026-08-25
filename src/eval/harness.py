"""Reproducible product-discovery evaluation harness.

Owner: Benyamin. The harness runs for free with MockRetriever, then switches to
the real retriever through the existing factory without changing the report
schema. Gold retrieval metrics activate only when gold product IDs are present.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.chains.product_discovery import ProductDiscoveryChain
from src.chains.product_filters import (
    LLMFilterExtractor,
    ProductFilterPlan,
    RuleBasedFilterExtractor,
)
from src.data.normalize import normalize
from src.eval.grounding import (
    CitationAudit,
    GroundingJudge,
    GroundingJudgeRun,
    LLMGroundingJudge,
    audit_citations,
)
from src.eval.retrieval_metrics import (
    load_qrels,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.llm.client import build_openai_client
from src.llm.config import LLMSettings
from src.llm.usage import SQLiteUsageLedger
from src.retrieval.base import Evidence, build_retriever


class EvalQuery(BaseModel):
    model_config = ConfigDict(extra="allow")

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    sub_cat: str | None = None
    relevant_product_ids: list[str] = Field(default_factory=list)
    relevance_judgements: dict[str, int] = Field(default_factory=dict)

    @field_validator("relevant_product_ids", mode="before")
    @classmethod
    def stringify_product_ids(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise ValueError("relevant_product_ids must be a list")
        return list(dict.fromkeys(str(item) for item in value))

    @field_validator("relevance_judgements", mode="before")
    @classmethod
    def normalize_relevance_judgements(cls, value: Any) -> dict[str, int]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("relevance_judgements must be an object")
        return {str(product_id): int(relevance) for product_id, relevance in value.items()}


class ConstraintAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checked_assertions: int = 0
    passed_assertions: int = 0
    uncheckable_assertions: int = 0
    violations: list[str] = Field(default_factory=list)


class RetrievalScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None
    retrieved_at_k: int = 0
    judged_retrieved_at_k: int = 0
    judgement_coverage_at_k: float | None = Field(default=None, ge=0, le=1)


class DiscoveryEvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    intent: str
    expected_sub_cat: str | None
    relevant_product_ids: list[str]
    retrieved_product_ids: list[str]
    result_count: int
    answer: str | None
    filter_plan: dict[str, Any] | None
    chain_latency_ms: float
    end_to_end_latency_ms: float
    category_known_results: int = 0
    category_matching_results: int = 0
    constraint_audit: ConstraintAudit
    citation_audit: CitationAudit | None = None
    retrieval_scores: RetrievalScores
    grounding_judgment: dict[str, Any] | None = None
    judge_error: str | None = None
    error: str | None = None


class DiscoveryEvaluator:
    def __init__(
        self,
        *,
        chain: ProductDiscoveryChain,
        top_k: int = 10,
        ledger: SQLiteUsageLedger | None = None,
        grounding_judge: GroundingJudge | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.chain = chain
        self.top_k = top_k
        self.ledger = ledger
        self.grounding_judge = grounding_judge

    def evaluate(self, queries: list[EvalQuery]) -> dict[str, Any]:
        if not queries:
            raise ValueError("evaluation query set cannot be empty")

        ledger_checkpoint = self.ledger.checkpoint() if self.ledger else 0
        run_started = time.perf_counter()
        items = [self._evaluate_one(query) for query in queries]
        wall_time_ms = (time.perf_counter() - run_started) * 1000

        return {
            "schema_version": "discovery-eval-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration": {
                "top_k": self.top_k,
                "retriever": type(self.chain.retriever).__name__,
                "filter_extractor": type(self.chain.extractor).__name__,
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

    def _evaluate_one(self, query: EvalQuery) -> DiscoveryEvalItem:
        item_started = time.perf_counter()
        chain_started = time.perf_counter()
        try:
            result = self.chain.run(query.query, top_k=self.top_k)
        except Exception as exc:  # keep one bad query from discarding the run
            chain_latency_ms = (time.perf_counter() - chain_started) * 1000
            return DiscoveryEvalItem(
                query_id=query.query_id,
                query=query.query,
                intent=query.intent,
                expected_sub_cat=query.sub_cat,
                relevant_product_ids=query.relevant_product_ids,
                retrieved_product_ids=[],
                result_count=0,
                answer=None,
                filter_plan=None,
                chain_latency_ms=chain_latency_ms,
                end_to_end_latency_ms=(time.perf_counter() - item_started) * 1000,
                constraint_audit=ConstraintAudit(),
                retrieval_scores=RetrievalScores(),
                error=f"{type(exc).__name__}: {exc}",
            )

        chain_latency_ms = (time.perf_counter() - chain_started) * 1000
        answer = result.render_fa()
        category_known, category_matching = _category_counts(
            result.products,
            query.sub_cat,
        )
        constraint_audit = _audit_constraints(result.products, result.filter_plan)
        citation_audit = audit_citations(answer, result.products)
        retrieval_scores = _retrieval_scores(
            result.products,
            query.relevant_product_ids,
            self.top_k,
            query.relevance_judgements,
        )

        grounding_run: GroundingJudgeRun | None = None
        judge_error: str | None = None
        if self.grounding_judge is not None:
            try:
                grounding_run = self.grounding_judge.evaluate(
                    question=query.query,
                    answer=answer,
                    evidence=result.products,
                )
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: {exc}"

        return DiscoveryEvalItem(
            query_id=query.query_id,
            query=query.query,
            intent=query.intent,
            expected_sub_cat=query.sub_cat,
            relevant_product_ids=query.relevant_product_ids,
            retrieved_product_ids=[item.id for item in result.products],
            result_count=len(result.products),
            answer=answer,
            filter_plan=result.filter_plan.model_dump(mode="json"),
            chain_latency_ms=chain_latency_ms,
            end_to_end_latency_ms=(time.perf_counter() - item_started) * 1000,
            category_known_results=category_known,
            category_matching_results=category_matching,
            constraint_audit=constraint_audit,
            citation_audit=citation_audit,
            retrieval_scores=retrieval_scores,
            grounding_judgment=(grounding_run.as_dict() if grounding_run else None),
            judge_error=judge_error,
        )


def load_eval_queries(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                query = EvalQuery.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid JSONL record at line {line_number}: {exc}") from exc
            if query.query_id in seen_ids:
                raise ValueError(f"duplicate query_id: {query.query_id}")
            seen_ids.add(query.query_id)
            queries.append(query)
    if not queries:
        raise ValueError(f"no evaluation queries found in {path}")
    return queries


def attach_qrels(
    queries: list[EvalQuery],
    qrels_path: Path,
    *,
    min_relevance: int = 1,
) -> list[EvalQuery]:
    """Attach Ali's separate graded qrels file to Benyamin's harness queries."""

    if min_relevance < 1:
        raise ValueError("min_relevance must be at least 1")
    qrels = load_qrels(qrels_path)
    query_ids = {query.query_id for query in queries}
    if not query_ids.intersection(qrels):
        raise ValueError("qrels file has no query_id in common with the eval set")

    attached: list[EvalQuery] = []
    for query in queries:
        judgements = qrels.get(query.query_id, {})
        relevant_ids = [
            product_id
            for product_id, relevance in judgements.items()
            if relevance >= min_relevance
        ]
        attached.append(
            query.model_copy(
                update={
                    "relevant_product_ids": relevant_ids,
                    "relevance_judgements": judgements,
                }
            )
        )
    return attached


def _category_counts(
    evidence: list[Evidence],
    expected_sub_cat: str | None,
) -> tuple[int, int]:
    if not expected_sub_cat:
        return 0, 0
    expected = normalize(expected_sub_cat)
    known = 0
    matching = 0
    for item in evidence:
        actual = item.meta.get("sub_cat")
        if actual in (None, ""):
            continue
        known += 1
        matching += int(normalize(str(actual)) == expected)
    return known, matching


def _audit_constraints(
    evidence: list[Evidence],
    plan: ProductFilterPlan,
) -> ConstraintAudit:
    checked = 0
    passed = 0
    uncheckable = 0
    violations: list[str] = []

    checks: list[tuple[str, str, Any, Any]] = [
        (
            "price_min",
            "price",
            plan.price_min_rial,
            lambda value: float(value) >= plan.price_min_rial,
        ),
        (
            "price_max",
            "price",
            plan.price_max_rial,
            lambda value: float(value) <= plan.price_max_rial,
        ),
        (
            "brand",
            "brand",
            plan.brands or None,
            lambda value: normalize(str(value))
            in {normalize(v) for v in plan.brands},
        ),
        (
            "cat1",
            "cat1",
            plan.cat1 or None,
            lambda value: normalize(str(value))
            in {normalize(v) for v in plan.cat1},
        ),
        (
            "sub_cat",
            "sub_cat",
            plan.sub_cat or None,
            lambda value: normalize(str(value))
            in {normalize(v) for v in plan.sub_cat},
        ),
        ("rate", "rate", plan.min_rate, lambda value: float(value) >= plan.min_rate),
        (
            "rate_count",
            "rate_count",
            plan.min_rate_count,
            lambda value: int(value) >= plan.min_rate_count,
        ),
        (
            "is_fake",
            "is_fake",
            True if plan.exclude_fake else None,
            lambda value: not _is_truthy(value),
        ),
    ]

    for item in evidence:
        for label, meta_field, active_value, predicate in checks:
            if active_value is None:
                continue
            value = item.meta.get(meta_field)
            if value is None and meta_field == "is_fake":
                value = item.meta.get("Is_Fake")
            if value is None:
                uncheckable += 1
                continue
            checked += 1
            try:
                is_valid = bool(predicate(value))
            except (TypeError, ValueError):
                is_valid = False
            if is_valid:
                passed += 1
            else:
                violations.append(f"{item.citation()}:{label}")

    return ConstraintAudit(
        checked_assertions=checked,
        passed_assertions=passed,
        uncheckable_assertions=uncheckable,
        violations=violations,
    )


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return normalize(value).lower() in {"1", "true", "yes", "بله"}
    return bool(value)


def _retrieval_scores(
    evidence: list[Evidence],
    relevant_product_ids: list[str],
    top_k: int,
    relevance_judgements: dict[str, int] | None = None,
) -> RetrievalScores:
    relevant = set(relevant_product_ids)
    retrieved = [item.product_id or item.id for item in evidence[:top_k]]
    if relevance_judgements:
        judged_retrieved = sum(
            identifier in relevance_judgements for identifier in retrieved
        )
        coverage = judged_retrieved / len(retrieved) if retrieved else None
        if not relevant:
            return RetrievalScores(
                retrieved_at_k=len(retrieved),
                judged_retrieved_at_k=judged_retrieved,
                judgement_coverage_at_k=coverage,
            )
        return RetrievalScores(
            recall_at_k=recall_at_k(retrieved, relevant, top_k),
            reciprocal_rank=mrr_at_k(retrieved, relevant, top_k),
            ndcg_at_k=ndcg_at_k(retrieved, relevance_judgements, top_k),
            retrieved_at_k=len(retrieved),
            judged_retrieved_at_k=judged_retrieved,
            judgement_coverage_at_k=coverage,
        )
    if not relevant:
        return RetrievalScores(retrieved_at_k=len(retrieved))
    seen_relevant: set[str] = set()
    hits: list[int] = []
    for identifier in retrieved:
        hit = identifier in relevant and identifier not in seen_relevant
        hits.append(int(hit))
        if hit:
            seen_relevant.add(identifier)
    recall = sum(hits) / len(relevant)
    first_hit = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    reciprocal_rank = 1 / first_hit if first_hit else 0.0
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), top_k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return RetrievalScores(
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=dcg / ideal_dcg if ideal_dcg else 0.0,
        retrieved_at_k=len(retrieved),
    )


def _summarize(
    items: list[DiscoveryEvalItem],
    wall_time_ms: float,
) -> dict[str, Any]:
    successful = [item for item in items if item.error is None]
    chain_latencies = [item.chain_latency_ms for item in successful]
    end_to_end_latencies = [item.end_to_end_latency_ms for item in successful]
    labeled = [
        item
        for item in successful
        if item.retrieval_scores.recall_at_k is not None
    ]
    coverage_items = [
        item
        for item in successful
        if item.retrieval_scores.judgement_coverage_at_k is not None
    ]
    judged = [
        item.grounding_judgment["judgment"]
        for item in successful
        if item.grounding_judgment is not None
    ]
    judged_claims = [
        claim
        for judgment in judged
        for claim in judgment["claims"]
    ]
    fully_supported_claims = [
        claim
        for claim in judged_claims
        if claim["verdict"] == "supported" and claim["evidence_ids"]
    ]

    category_known = sum(item.category_known_results for item in successful)
    category_matching = sum(item.category_matching_results for item in successful)
    constraint_checked = sum(
        item.constraint_audit.checked_assertions for item in successful
    )
    constraint_passed = sum(
        item.constraint_audit.passed_assertions for item in successful
    )
    citation_found = sum(
        len(item.citation_audit.found)
        for item in successful
        if item.citation_audit is not None
    )
    citation_valid = sum(
        len(item.citation_audit.valid)
        for item in successful
        if item.citation_audit is not None
    )
    retrieved_for_coverage = sum(
        item.retrieval_scores.retrieved_at_k for item in coverage_items
    )
    judged_retrieved = sum(
        item.retrieval_scores.judged_retrieved_at_k for item in coverage_items
    )
    judgement_coverage = (
        judged_retrieved / retrieved_for_coverage
        if retrieved_for_coverage
        else None
    )

    return {
        "query_count": len(items),
        "successful_queries": len(successful),
        "errored_queries": len(items) - len(successful),
        "empty_result_queries": sum(item.result_count == 0 for item in successful),
        "judge_errors": sum(item.judge_error is not None for item in successful),
        "wall_time_ms": wall_time_ms,
        "chain_latency_ms": _distribution(chain_latencies),
        "end_to_end_latency_ms": _distribution(end_to_end_latencies),
        "mean_result_count": (
            fmean(item.result_count for item in successful) if successful else 0.0
        ),
        "category_match_rate": (
            category_matching / category_known if category_known else None
        ),
        "constraint_pass_rate": (
            constraint_passed / constraint_checked if constraint_checked else None
        ),
        "citation_integrity_rate": (
            citation_valid / citation_found if citation_found else None
        ),
        "retrieval": {
            "available": bool(labeled),
            "labeled_queries": len(labeled),
            "unlabeled_queries": len(successful) - len(labeled),
            "mean_recall_at_k": _mean_optional(
                item.retrieval_scores.recall_at_k for item in labeled
            ),
            "mrr": _mean_optional(
                item.retrieval_scores.reciprocal_rank for item in labeled
            ),
            "mean_ndcg_at_k": _mean_optional(
                item.retrieval_scores.ndcg_at_k for item in labeled
            ),
            "judgement_coverage_at_k": judgement_coverage,
            "queries_with_zero_judged_results": sum(
                item.retrieval_scores.judged_retrieved_at_k == 0
                for item in coverage_items
            ),
            "note": _retrieval_note(bool(labeled), judgement_coverage),
        },
        "grounding_judge": {
            "judged_queries": len(judged),
            "mean_relevance_score": _mean_optional(
                float(item["relevance_score"]) for item in judged
            ),
            "mean_grounding_score": _mean_optional(
                float(item["grounding_score"]) for item in judged
            ),
            "fully_supported_claim_rate": (
                len(fully_supported_claims) / len(judged_claims)
                if judged_claims
                else None
            ),
            "verdict_counts": {
                verdict: sum(item["verdict"] == verdict for item in judged)
                for verdict in (
                    "grounded",
                    "partially_grounded",
                    "ungrounded",
                )
            },
        },
        "by_intent": _summarize_by_intent(successful),
    }


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


def _mean_optional(values: Any) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def _summarize_by_intent(items: list[DiscoveryEvalItem]) -> dict[str, Any]:
    buckets: dict[str, list[DiscoveryEvalItem]] = {}
    for item in items:
        buckets.setdefault(item.intent, []).append(item)
    return {
        intent: {
            "queries": len(bucket),
            "empty_result_queries": sum(item.result_count == 0 for item in bucket),
            "chain_latency_ms": _distribution(
                [item.chain_latency_ms for item in bucket]
            ),
        }
        for intent, bucket in sorted(buckets.items())
    }


def _retrieval_note(
    has_labels: bool,
    judgement_coverage: float | None,
) -> str | None:
    if not has_labels:
        return "Recall/MRR/nDCG require inline gold IDs or --qrels."
    if judgement_coverage is not None and judgement_coverage < 1:
        return (
            "Pooled qrels do not cover every retrieved result. Unjudged items "
            "are treated as non-relevant, so metrics can be strongly biased "
            "until this run is added to the judging pool."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate product discovery.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/eval/queries_v1.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--qrels",
        type=Path,
        help="Optional graded qrels CSV, e.g. data/eval/qrels_d50_v2_labeled.csv.",
    )
    parser.add_argument("--min-relevance", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retriever-mode", default="mock")
    parser.add_argument("--use-llm-filters", action="store_true")
    parser.add_argument("--judge-grounding", action="store_true")
    args = parser.parse_args()

    settings = LLMSettings.from_env()
    llm_client = (
        build_openai_client(settings)
        if args.use_llm_filters or args.judge_grounding
        else None
    )
    extractor = (
        LLMFilterExtractor(llm_client)
        if args.use_llm_filters and llm_client
        else RuleBasedFilterExtractor()
    )
    judge = (
        LLMGroundingJudge(llm_client)
        if args.judge_grounding and llm_client
        else None
    )
    ledger = SQLiteUsageLedger(settings.usage_path) if llm_client else None
    evaluator = DiscoveryEvaluator(
        chain=ProductDiscoveryChain(
            retriever=build_retriever("product", mode=args.retriever_mode),
            extractor=extractor,
        ),
        top_k=args.top_k,
        ledger=ledger,
        grounding_judge=judge,
    )
    queries = load_eval_queries(args.input)
    if args.qrels:
        queries = attach_qrels(
            queries,
            args.qrels,
            min_relevance=args.min_relevance,
        )
    report = evaluator.evaluate(queries)
    report["configuration"]["qrels"] = str(args.qrels) if args.qrels else None
    report["configuration"]["min_relevance"] = args.min_relevance
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
