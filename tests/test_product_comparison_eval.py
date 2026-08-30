from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.chains.product_comparison import (
    ComparisonInference,
    ProductComparisonChain,
    ProductComparisonResult,
    ProductFacts,
    ProductReviewEvidence,
)
from src.eval.product_comparison import (
    ComparisonEvalCase,
    ProductComparisonEvaluator,
    load_comparison_cases,
)
from src.eval.grounding import GroundingJudgeRun, GroundingJudgment
from src.retrieval.base import Evidence, RetrievalFilters, Retriever


class StaticRetriever(Retriever):
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:
        items = self.evidence
        if filters and filters.product_ids:
            product_ids = {str(product_id) for product_id in filters.product_ids}
            items = [
                item
                for item in items
                if str(item.product_id or item.id) in product_ids
            ]
        return items[:top_k]


class ResultChain:
    def __init__(self, result: ProductComparisonResult) -> None:
        self.result = result
        self.product_retriever = StaticRetriever([])
        self.comment_retriever = StaticRetriever([])
        self.comment_top_k = 5
        self.llm_client = object()

    def run(self, query: str, *, product_ids: list[str]) -> ProductComparisonResult:
        return self.result


class StaticJudge:
    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[Evidence],
    ) -> GroundingJudgeRun:
        return GroundingJudgeRun(
            judgment=GroundingJudgment(
                relevance_score=5,
                grounding_score=4,
                verdict="ungrounded",  # validator derives the canonical verdict
                rationale="supported by the supplied evidence",
            ),
            model="fake-judge",
            cache_hit=False,
            latency_ms=1.0,
        )


def _product(product_id: str) -> Evidence:
    return Evidence(
        id=product_id,
        kind="product",
        text=f"product {product_id}",
        score=1.0,
        product_id=product_id,
        title=f"product {product_id}",
        meta={"price": 100_000, "rate": 80, "rate_count": 10},
    )


def _comment(comment_id: str, product_id: str) -> Evidence:
    return Evidence(
        id=comment_id,
        kind="comment",
        text=f"review {comment_id}",
        score=0.9,
        product_id=product_id,
    )


def _case() -> ComparisonEvalCase:
    return ComparisonEvalCase(
        case_id="c1",
        pair_id="pair-1",
        category="test",
        query="این دو محصول را مقایسه کن",
        product_ids=["p1", "p2"],
    )


class ProductComparisonEvaluatorTests(unittest.TestCase):
    def test_retrieval_only_pass_is_not_reported_as_quality(self) -> None:
        chain = ProductComparisonChain(
            product_retriever=StaticRetriever([_product("p1"), _product("p2")]),
            comment_retriever=StaticRetriever(
                [_comment("c1", "p1"), _comment("c2", "p2")]
            ),
            llm_client=None,
        )

        report = ProductComparisonEvaluator(chain=chain).evaluate([_case()])

        self.assertEqual(report["summary"]["structural_passed"], 1)
        self.assertEqual(report["summary"]["inference_available"], 0)
        self.assertEqual(
            report["summary"]["quality_assessment"]["status"],
            "not_measured",
        )
        self.assertTrue(
            report["items"][0]["structural_audit"]["product_scope_consistent"]
        )

    def test_structural_pass_requires_evidence_for_both_products(self) -> None:
        chain = ProductComparisonChain(
            product_retriever=StaticRetriever([_product("p1"), _product("p2")]),
            comment_retriever=StaticRetriever([_comment("c1", "p1")]),
            llm_client=None,
        )

        item = ProductComparisonEvaluator(chain=chain).evaluate([_case()])["items"][0]

        self.assertFalse(item["structural_audit"]["all_products_have_evidence"])
        self.assertFalse(item["structural_audit"]["structural_pass"])
        self.assertEqual(item["structural_audit"]["evidence_counts"]["p2"], 0)

    def test_inline_invented_inference_citation_is_visible_in_report(self) -> None:
        facts = [
            ProductFacts.from_evidence(_product("p1")),
            ProductFacts.from_evidence(_product("p2")),
        ]
        result = ProductComparisonResult(
            query=_case().query,
            requested_product_ids=["p1", "p2"],
            facts=facts,
            evidence=[
                ProductReviewEvidence("p1", "q", [_comment("c1", "p1")]),
                ProductReviewEvidence("p2", "q", [_comment("c2", "p2")]),
            ],
            inference=ComparisonInference(
                overall_recommendation=(
                    "محصول اول بهتر است [comment:c1] اما این شاهد ساختگی است "
                    "[comment:ghost]."
                ),
                citations=["[comment:c1]"],
            ),
            missing_products=[],
            warnings=[],
        )

        report = ProductComparisonEvaluator(chain=ResultChain(result)).evaluate([_case()])

        audit = report["items"][0]["inference_citation_audit"]
        self.assertEqual(audit["invalid"], ["[comment:ghost]"])
        self.assertEqual(report["summary"]["citation_integrity"]["inference_invalid_ids"], 1)
        self.assertEqual(
            report["summary"]["quality_assessment"]["status"],
            "citation_only",
        )

    def test_judge_scores_are_aggregated_separately(self) -> None:
        facts = [
            ProductFacts.from_evidence(_product("p1")),
            ProductFacts.from_evidence(_product("p2")),
        ]
        result = ProductComparisonResult(
            query=_case().query,
            requested_product_ids=["p1", "p2"],
            facts=facts,
            evidence=[
                ProductReviewEvidence("p1", "q", [_comment("c1", "p1")]),
                ProductReviewEvidence("p2", "q", [_comment("c2", "p2")]),
            ],
            inference=ComparisonInference(
                overall_recommendation="محصول اول بهتر است [comment:c1].",
                citations=["[comment:c1]"],
            ),
            missing_products=[],
            warnings=[],
        )

        report = ProductComparisonEvaluator(
            chain=ResultChain(result),
            grounding_judge=StaticJudge(),
        ).evaluate([_case()])

        self.assertEqual(report["summary"]["grounding_judged"], 1)
        self.assertEqual(report["summary"]["judge_scores"]["mean_grounding"], 4)
        self.assertEqual(report["summary"]["judge_scores"]["mean_relevance"], 5)
        self.assertEqual(
            report["summary"]["judge_scores"]["verdict_counts"]["grounded"],
            1,
        )
        self.assertEqual(
            report["summary"]["quality_assessment"]["status"],
            "judge_measured",
        )

    def test_loader_rejects_duplicate_case_ids(self) -> None:
        record = _case().model_dump_json()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(f"{record}\n{record}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                load_comparison_cases(path)

    def test_case_requires_distinct_product_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            ComparisonEvalCase(
                case_id="c1",
                pair_id="pair-1",
                query="compare",
                product_ids=["p1", "p1"],
            )


if __name__ == "__main__":
    unittest.main()
