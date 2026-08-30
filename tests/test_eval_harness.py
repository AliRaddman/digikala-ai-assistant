"""Offline tests for Benyamin's evaluation harness and grounding judge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.chains.product_discovery import ProductDiscoveryChain
from src.chains.product_filters import RuleBasedFilterExtractor
from src.eval.grounding import LLMGroundingJudge, audit_citations
from src.eval.harness import DiscoveryEvaluator, EvalQuery, attach_qrels
from src.llm.cache import SQLiteLLMCache
from src.llm.client import CachedLLMClient, ProviderResult
from src.llm.types import TokenUsage
from src.llm.usage import SQLiteUsageLedger
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
        return self.evidence[:top_k]


class StaticGroundingProvider:
    def __init__(self, evidence_id: str = "[product:p1]") -> None:
        self.evidence_id = evidence_id
        self.calls = 0
        # Overridable so a test can reproduce the score/verdict pair that the
        # live model sent and the old validator rejected.
        self.grounding_score = 5
        self.verdict = "grounded"

    def generate_structured(self, *, model, messages, response_model):
        self.calls += 1
        return ProviderResult(
            data={
                "relevance_score": 5,
                "grounding_score": self.grounding_score,
                "verdict": self.verdict,
                "claims": [
                    {
                        "claim": "قیمت محصول ۵۰ هزار تومان است.",
                        "verdict": "supported",
                        "evidence_ids": [self.evidence_id],
                        "rationale": "قیمت در متادیتای شاهد آمده است.",
                    }
                ],
                "rationale": "پاسخ مستقیماً به شاهد متکی است.",
            },
            model=model,
            request_id="judge_test",
            usage=TokenUsage(input_tokens=500, output_tokens=100),
        )


def _products() -> list[Evidence]:
    return [
        Evidence(
            id="p1",
            kind="product",
            product_id="p1",
            title="کیف ارزان",
            text="کیف ارزان",
            score=0.9,
            meta={
                "price": 500_000,
                "rate": 80,
                "rate_count": 10,
                "sub_cat": "clothe",
            },
        ),
        Evidence(
            id="p2",
            kind="product",
            product_id="p2",
            title="کیف گران",
            text="کیف گران",
            score=0.8,
            meta={
                "price": 2_000_000,
                "rate": 90,
                "rate_count": 20,
                "sub_cat": "clothe",
            },
        ),
    ]


class EvaluationHarnessTests(unittest.TestCase):
    def test_ali_qrels_are_attached_by_query_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qrels.csv"
            path.write_text(
                "query_id,product_id,relevance\n"
                "q1,p1,2\n"
                "q1,p2,0\n"
                "q2,p3,1\n",
                encoding="utf-8",
            )
            queries = [
                EvalQuery(query_id="q1", query="کیف", intent="simple"),
                EvalQuery(query_id="q2", query="کتاب", intent="simple"),
            ]

            attached = attach_qrels(queries, path)

            self.assertEqual(attached[0].relevant_product_ids, ["p1"])
            self.assertEqual(attached[0].relevance_judgements, {"p1": 2, "p2": 0})
            self.assertEqual(attached[1].relevant_product_ids, ["p3"])

    def test_citation_audit_separates_unknown_ids(self) -> None:
        audit = audit_citations(
            "خوب است [product:p1] ولی شاهد دوم جعلی است [comment:missing]",
            _products(),
        )

        self.assertEqual(audit.valid, ["[product:p1]"])
        self.assertEqual(audit.invalid, ["[comment:missing]"])
        self.assertEqual(audit.integrity_score, 0.5)

    def test_grounding_judge_is_structured_cached_and_evidence_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = StaticGroundingProvider()
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
            judge = LLMGroundingJudge(client)
            kwargs = {
                "question": "قیمت چقدر است؟",
                "answer": "قیمت ۵۰ هزار تومان است [product:p1]",
                "evidence": [_products()[0]],
            }

            first = judge.evaluate(**kwargs)
            second = judge.evaluate(**kwargs)

            self.assertEqual(first.judgment.grounding_score, 5)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(provider.calls, 1)

    def test_grounding_judge_rejects_an_unknown_evidence_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = CachedLLMClient(
                provider=StaticGroundingProvider("[product:missing]"),
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )

            with self.assertRaisesRegex(ValueError, "unknown evidence ids"):
                LLMGroundingJudge(client).evaluate(
                    question="قیمت چقدر است؟",
                    answer="پاسخ [product:p1]",
                    evidence=[_products()[0]],
                )

    def test_harness_reports_gold_metrics_and_filter_violations(self) -> None:
        evaluator = DiscoveryEvaluator(
            chain=ProductDiscoveryChain(
                retriever=StaticRetriever(_products()),
                extractor=RuleBasedFilterExtractor(),
            ),
            top_k=2,
        )
        report = evaluator.evaluate(
            [
                EvalQuery(
                    query_id="q1",
                    query="کیف زیر ۱۰۰ هزار تومان",
                    intent="price",
                    sub_cat="clothe",
                    relevant_product_ids=["p2"],
                )
            ]
        )

        item = report["items"][0]
        self.assertEqual(item["constraint_audit"]["checked_assertions"], 2)
        self.assertEqual(item["constraint_audit"]["passed_assertions"], 1)
        self.assertEqual(item["constraint_audit"]["violations"], ["[product:p2]:price_max"])
        self.assertEqual(item["retrieval_scores"]["recall_at_k"], 1.0)
        self.assertEqual(item["retrieval_scores"]["reciprocal_rank"], 0.5)
        self.assertTrue(report["summary"]["retrieval"]["available"])
        self.assertEqual(report["summary"]["category_match_rate"], 1.0)

    def test_harness_warns_when_pooled_qrels_do_not_cover_a_new_run(self) -> None:
        evaluator = DiscoveryEvaluator(
            chain=ProductDiscoveryChain(
                retriever=StaticRetriever(_products()),
                extractor=RuleBasedFilterExtractor(),
            ),
            top_k=2,
        )
        report = evaluator.evaluate(
            [
                EvalQuery(
                    query_id="q1",
                    query="کیف",
                    intent="simple",
                    relevant_product_ids=["p1"],
                    relevance_judgements={"p1": 2},
                )
            ]
        )

        retrieval = report["summary"]["retrieval"]
        self.assertEqual(retrieval["judgement_coverage_at_k"], 0.5)
        self.assertIn("Pooled qrels", retrieval["note"])

    def test_harness_integrates_judge_and_scoped_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = SQLiteUsageLedger(root / "usage.sqlite3")
            client = CachedLLMClient(
                provider=StaticGroundingProvider(),
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=ledger,
            )
            evaluator = DiscoveryEvaluator(
                chain=ProductDiscoveryChain(
                    retriever=StaticRetriever([_products()[0]]),
                    extractor=RuleBasedFilterExtractor(),
                ),
                top_k=1,
                ledger=ledger,
                grounding_judge=LLMGroundingJudge(client),
            )

            report = evaluator.evaluate(
                [EvalQuery(query_id="q1", query="کیف", intent="simple")]
            )

            grounding = report["summary"]["grounding_judge"]
            self.assertEqual(grounding["judged_queries"], 1)
            self.assertEqual(grounding["mean_grounding_score"], 5.0)
            self.assertEqual(grounding["fully_supported_claim_rate"], 1.0)
            self.assertEqual(report["llm_usage"]["logical_requests"], 1)
            self.assertEqual(report["llm_usage"]["api_calls"], 1)

    def test_usage_summary_is_scoped_to_a_checkpoint_and_has_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = SQLiteUsageLedger(Path(directory) / "usage.sqlite3")
            ledger.record(
                operation="old",
                model="gpt-4o-mini",
                request_id="old",
                cache_hit=False,
                usage=TokenUsage(input_tokens=100),
                latency_ms=999,
                cost_usd=0.1,
            )
            checkpoint = ledger.checkpoint()
            for latency, cache_hit in [(10, False), (20, True), (100, False)]:
                ledger.record(
                    operation="eval",
                    model="gpt-4o-mini",
                    request_id=None,
                    cache_hit=cache_hit,
                    usage=TokenUsage(input_tokens=10),
                    latency_ms=latency,
                    cost_usd=0.0,
                )

            summary = ledger.summary(after_id=checkpoint, operation="eval")

            self.assertEqual(summary["logical_requests"], 3)
            self.assertEqual(summary["api_calls"], 2)
            self.assertEqual(summary["cache_hits"], 1)
            self.assertEqual(summary["p50_latency_ms"], 20)
            self.assertAlmostEqual(summary["p95_latency_ms"], 92)


class GroundingJudgeLiveRegressionTests(unittest.TestCase):
    """Regressions for the two judge bugs the first live run exposed.

    Added by Ali on 2026-08-30. StaticGroundingProvider above returns exactly
    what the validators wanted, which is why the offline suite stayed green
    while all 36 real calls failed. These fakes reproduce what gpt-4o-mini
    actually sent instead. See the docstring of src/eval/grounding.py.
    """

    @staticmethod
    def _judge(provider: object, root: Path) -> LLMGroundingJudge:
        return LLMGroundingJudge(
            CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
        )

    def test_judge_accepts_a_valid_evidence_id_written_without_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._judge(
                StaticGroundingProvider("product:p1"), Path(directory)
            ).evaluate(
                question="قیمت چقدر است؟",
                answer="قیمت ۵۰ هزار تومان است [product:p1]",
                evidence=[_products()[0]],
            )

            self.assertEqual(run.judgment.claims[0].evidence_ids, ["product:p1"])

    def test_judge_still_rejects_an_unknown_id_written_without_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            judge = self._judge(
                StaticGroundingProvider("product:missing"), Path(directory)
            )

            with self.assertRaisesRegex(ValueError, "unknown evidence ids"):
                judge.evaluate(
                    question="قیمت چقدر است؟",
                    answer="پاسخ [product:p1]",
                    evidence=[_products()[0]],
                )

    def test_judge_derives_the_verdict_instead_of_rejecting_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = StaticGroundingProvider("[product:p1]")
            provider.grounding_score = 4
            provider.verdict = "partially_grounded"

            run = self._judge(provider, Path(directory)).evaluate(
                question="قیمت چقدر است؟",
                answer="قیمت ۵۰ هزار تومان است [product:p1]",
                evidence=[_products()[0]],
            )

            self.assertEqual(run.judgment.grounding_score, 4)
            self.assertEqual(run.judgment.verdict, "grounded")


if __name__ == "__main__":
    unittest.main()
