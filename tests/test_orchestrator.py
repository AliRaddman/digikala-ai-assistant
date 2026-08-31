"""Offline tests for Benyamin's chain orchestrator."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.chains.category_analytics import load_products
from src.llm.cache import SQLiteLLMCache
from src.llm.client import CachedLLMClient, ProviderResult
from src.llm.types import TokenUsage
from src.llm.usage import SQLiteUsageLedger
from src.orchestrator import (
    CategoryAnalyticsHandler,
    HandlerResult,
    OrchestratorRequest,
    ProductComparisonHandler,
    ProductQAHandler,
    RuleBasedIntentRouter,
    ShoppingAssistantOrchestrator,
    build_default_orchestrator,
    extract_product_ids,
)
from src.retrieval.base import MockRetriever
from src.retrieval.comments import CommentRetriever


class FakeQAProvider:
    """Stands in for the OpenAI provider: always answers with one grounded
    claim citing the first comment_id it was shown as evidence."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(self, *, model, messages, response_model):
        self.calls += 1
        return ProviderResult(
            data={
                "answer_fa": "کاربران به زیپ ضعیف اشاره کرده‌اند.",
                "sufficient_evidence": True,
                "claims": [
                    {
                        "text": "زیپ بعد از مدتی خراب می‌شود.",
                        "comment_ids": ["51230044"],
                    }
                ],
            },
            model=model,
            request_id="resp_test",
            usage=TokenUsage(input_tokens=200, output_tokens=40),
        )


class RecordingHandler:
    def __init__(self) -> None:
        self.requests: list[OrchestratorRequest] = []

    def handle(self, request: OrchestratorRequest) -> HandlerResult:
        self.requests.append(request)
        return HandlerResult(
            answer="مقایسه آزمایشی",
            citations=["[product:10]", "[product:20]"],
            payload={"product_ids": request.product_ids},
        )


class FailingHandler:
    def handle(self, request: OrchestratorRequest) -> HandlerResult:
        raise RuntimeError("boom")


class FakeComparisonProvider:
    """Stands in for the OpenAI provider on the comparison path."""

    def __init__(self) -> None:
        self.calls = 0
        # a real model ignores the bracket convention often enough that both
        # spellings have to be exercised; see _canonical_citation
        self.bare_citations = False
        self.extra_citation: str | None = None

    def _tag(self, citation: str) -> str:
        return citation.strip("[]") if self.bare_citations else citation

    def generate_structured(self, *, model, messages, response_model):
        self.calls += 1
        return ProviderResult(
            data={
                "product_assessments": [
                    {
                        "product_id": "3901234",
                        "strengths": ["قیمت مناسب"],
                        "weaknesses": ["زیپ ضعیف"],
                        "citations": [self._tag("[comment:51230044]")],
                    }
                ],
                "criteria": [
                    {
                        "criterion": "کیفیت ساخت",
                        "better_product_id": "6604311",
                        "explanation": "نظرات این محصول از دوام بیشتری می‌گویند.",
                        "citations": [self._tag("[comment:60020155]")],
                    }
                ],
                "overall_winner_product_id": "6604311",
                "overall_recommendation": "با توجه به نظرات، محصول دوم انتخاب بهتری است.",
                "caveats": ["تعداد نظرات کم است."],
                "citations": [
                    self._tag("[product:3901234]"),
                    self._tag("[product:6604311]"),
                    *([self.extra_citation] if self.extra_citation else []),
                ],
            },
            model=model,
            request_id="resp_comparison_test",
            usage=TokenUsage(input_tokens=400, output_tokens=90),
        )


class ProductComparisonRouteTests(unittest.TestCase):
    """The comparison path answers, rather than merely being registered.

    Ali, 2026-08-30. Fatemeh's chain existed but was never added to
    build_default_orchestrator, so every comparison request came back as
    dependency_unavailable ['product_comparison_chain'].
    """

    @staticmethod
    def _orchestrator(client: CachedLLMClient | None) -> ShoppingAssistantOrchestrator:
        return ShoppingAssistantOrchestrator(
            router=RuleBasedIntentRouter(),
            handlers={
                "product_comparison": ProductComparisonHandler(
                    product_retriever=MockRetriever("product"),
                    comment_retriever=MockRetriever("comment"),
                    client=client,
                )
            },
        )

    def test_comparison_answers_end_to_end_with_a_fake_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeComparisonProvider()
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
            result = self._orchestrator(client).run(
                "این دو محصول را مقایسه کن",
                product_ids=["3901234", "6604311"],
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.route.intent, "product_comparison")
        self.assertEqual(provider.calls, 1)
        # both products resolved, so neither is reported missing
        self.assertEqual(result.payload["missing_products"], [])
        self.assertIn("[product:3901234]", result.answer)
        self.assertIn("[product:6604311]", result.answer)
        self.assertIn("جمع‌بندی استنباطی", result.answer)

    def test_comparison_fills_the_review_evidence_layer(self) -> None:
        """Fatemeh's notebook left this empty; the comment index exists now."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = CachedLLMClient(
                provider=FakeComparisonProvider(),
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
            result = self._orchestrator(client).run(
                "این دو محصول را مقایسه کن",
                product_ids=["3901234", "6604311"],
            )

        evidence = result.payload["evidence"]
        self.assertTrue(evidence["available"])
        by_product = {group["product_id"]: group for group in evidence["items"]}
        self.assertTrue(by_product["3901234"]["items"])
        self.assertTrue(by_product["6604311"]["items"])
        # every retrieved comment belongs to the product it was gathered for
        for product_id, group in by_product.items():
            for comment in group["items"]:
                self.assertEqual(comment["product_id"], product_id)
        self.assertTrue(any(c.startswith("[comment:") for c in result.citations))

    def test_comparison_still_answers_without_an_llm_key(self) -> None:
        """Facts and evidence need no model; only the inference block is lost."""
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
            result = self._orchestrator(None).run(
                "این دو محصول را مقایسه کن",
                product_ids=["3901234", "6604311"],
            )

        self.assertEqual(result.status, "success")
        self.assertIsNone(result.payload["inference"])
        self.assertIn("[product:3901234]", result.answer)
        self.assertIn("استنباط LLM غیرفعال است", result.answer)

    def test_a_citation_without_brackets_is_kept_not_silently_dropped(self) -> None:
        """The live-model format that destroyed 34 judgments in the judge."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeComparisonProvider()
            # what gpt-4o-mini actually emits from a bracketed instruction
            provider.bare_citations = True
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
            result = self._orchestrator(client).run(
                "این دو محصول را مقایسه کن",
                product_ids=["3901234", "6604311"],
            )

        inference = result.payload["inference"]
        self.assertEqual(
            inference["citations"], ["[product:3901234]", "[product:6604311]"]
        )
        self.assertEqual(
            inference["product_assessments"][0]["citations"], ["[comment:51230044]"]
        )

    def test_an_invented_citation_is_still_dropped_without_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeComparisonProvider()
            provider.bare_citations = True
            provider.extra_citation = "comment:00000000"
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
            result = self._orchestrator(client).run(
                "این دو محصول را مقایسه کن",
                product_ids=["3901234", "6604311"],
            )

        citations = result.payload["inference"]["citations"]
        self.assertNotIn("[comment:00000000]", citations)
        self.assertIn("[product:3901234]", citations)

    def test_default_orchestrator_registers_the_comparison_handler(self) -> None:
        orchestrator = build_default_orchestrator()

        self.assertIn("product_comparison", orchestrator.handlers)
        result = orchestrator.run(
            "این دو محصول را مقایسه کن",
            product_ids=["3901234", "6604311"],
        )
        self.assertNotEqual(result.status, "dependency_unavailable")


class CategoryAnalyticsRoutingTests(unittest.TestCase):
    """The handler side of section-4 question routing (Ali, 2026-08-30)."""

    @staticmethod
    def _fixture(root: Path):
        products_path = root / "products.parquet"
        comments_path = root / "comments.parquet"
        pd.DataFrame(
            [
                {"product_id": "1", "cat1": "اسباب بازی", "cat2": None,
                 "sub_cat": "toy", "brand": "الف"},
                {"product_id": "2", "cat1": "اسباب بازی", "cat2": None,
                 "sub_cat": "toy", "brand": "ب"},
            ]
        ).to_parquet(products_path, index=False)
        pd.DataFrame(
            [
                {"product_id": "1", "disadvantages": "قیمت بالا",
                 "recommendation_status": "not_recommended", "rate": 2.0},
                {"product_id": "2", "disadvantages": "کیفیت پایین",
                 "recommendation_status": "recommended", "rate": 5.0},
            ]
        ).to_parquet(comments_path, index=False)
        return ShoppingAssistantOrchestrator(
            router=RuleBasedIntentRouter(),
            handlers={
                "category_analytics": CategoryAnalyticsHandler(
                    products_path=products_path,
                    comments_path=comments_path,
                    client=None,
                )
            },
        )

    def test_the_category_comes_from_context_when_the_question_omits_it(self) -> None:
        """All four brief examples say "این دسته" and name no category."""
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._fixture(Path(directory))
            load_products.cache_clear()
            with mock.patch.dict(os.environ, {}, clear=True):
                result = orchestrator.run(
                    "پرتکرارترین شکایت کاربران در این دسته چیست؟",
                    context_product_ids=["2"],
                )
            load_products.cache_clear()

        self.assertEqual(result.status, "success")
        self.assertEqual(result.payload["scope"]["cat1"], "اسباب بازی")
        self.assertEqual(result.payload["question"], "top_complaints")

    def test_the_brand_question_returns_the_brand_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._fixture(Path(directory))
            load_products.cache_clear()
            with mock.patch.dict(os.environ, {}, clear=True):
                result = orchestrator.run(
                    "بازخورد کاربران درباره‌ی برندهای دسته اسباب بازی چه تفاوتی دارد؟"
                )
            load_products.cache_clear()

        self.assertEqual(result.payload["question"], "brand_feedback")
        self.assertIn("برند", result.answer)

    def test_an_unresolvable_category_offers_real_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = self._fixture(Path(directory))
            load_products.cache_clear()
            with mock.patch.dict(os.environ, {}, clear=True):
                result = orchestrator.run("پرتکرارترین شکایت کاربران در این دسته چیست؟")
            load_products.cache_clear()

        # a real category from the catalogue, not a vague apology
        self.assertIn("اسباب بازی", result.answer)


class OrchestratorTests(unittest.TestCase):
    def test_product_ids_are_extracted_without_treating_price_as_an_id(self) -> None:
        query = "محصول [product:۱۲۳۴۵] را با شناسه محصول: 67890 زیر ۵۰۰۰۰۰ تومان مقایسه کن"
        self.assertEqual(extract_product_ids(query), ["12345", "67890"])

    def test_discovery_with_satisfaction_language_does_not_route_to_qa(self) -> None:
        decision = RuleBasedIntentRouter().route(
            OrchestratorRequest(
                query="یه کیف می‌خوام که خریدارها ازش راضی باشن"
            )
        )
        self.assertEqual(decision.intent, "product_discovery")
        self.assertEqual(decision.confidence, 0.95)

    def test_old_context_does_not_hijack_an_explicit_discovery_request(self) -> None:
        decision = RuleBasedIntentRouter().route(
            OrchestratorRequest(
                query="یک کیف روزمره معرفی کن",
                context_product_ids=["10", "20"],
            )
        )
        self.assertEqual(decision.intent, "product_discovery")

    def test_context_can_resolve_a_deictic_product_qa_request(self) -> None:
        decision = RuleBasedIntentRouter().route(
            OrchestratorRequest(
                query="ایرادهای پرتکرار این محصول چیست؟",
                context_product_ids=["3901234"],
            )
        )
        self.assertEqual(decision.intent, "product_qa")
        self.assertEqual(decision.product_ids, ["3901234"])

    def test_category_question_has_precedence_over_product_qa_words(self) -> None:
        decision = RuleBasedIntentRouter().route(
            OrchestratorRequest(
                query="پرتکرارترین شکایت کاربران در این دسته چیست؟"
            )
        )
        self.assertEqual(decision.intent, "category_analytics")

    def test_comparison_without_two_ids_requests_missing_input(self) -> None:
        result = build_default_orchestrator().run(
            "این دو محصول را مقایسه کن؛ کدام بهتر است؟",
            product_ids=["10"],
        )
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(result.route.intent, "product_comparison")
        self.assertEqual(result.missing_requirements, ["at_least_two_product_ids"])

    def test_product_qa_answers_end_to_end_with_a_fake_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeQAProvider()
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
            )
            orchestrator = ShoppingAssistantOrchestrator(
                router=RuleBasedIntentRouter(),
                handlers={
                    "product_qa": ProductQAHandler(
                        retriever=MockRetriever("comment"), client=client
                    )
                },
            )
            result = orchestrator.run(
                "ایرادهای پرتکرار این محصول چیست؟",
                product_ids=["3901234"],
            )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.route.intent, "product_qa")
        self.assertTrue(result.citations)
        self.assertIn("[comment:", result.answer)
        self.assertEqual(provider.calls, 1)

    def test_product_qa_reports_a_clean_error_when_the_comment_index_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as empty_index_dir:
            orchestrator = ShoppingAssistantOrchestrator(
                router=RuleBasedIntentRouter(),
                handlers={
                    "product_qa": ProductQAHandler(
                        retriever=CommentRetriever(index_dir=Path(empty_index_dir))
                    )
                },
            )
            result = orchestrator.run(
                "ایرادهای پرتکرار این محصول چیست؟",
                product_ids=["3901234"],
            )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.route.intent, "product_qa")
        self.assertIsNotNone(result.error)

    def test_default_orchestrator_runs_product_discovery_end_to_end(self) -> None:
        result = build_default_orchestrator().run(
            "یک کیف روزمره زیر ۲۰۰ هزار تومان معرفی کن",
            top_k=3,
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.route.intent, "product_discovery")
        self.assertTrue(result.citations)
        self.assertIn("محصول‌های پیشنهادی", result.answer)

    def test_category_analytics_still_answers_without_an_llm_key(self) -> None:
        """The aggregation is the answer; the model only narrates it, so a
        missing API key must degrade to the computed tables, not fail."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products_path = root / "products.parquet"
            comments_path = root / "comments.parquet"
            pd.DataFrame(
                [{"product_id": "1", "cat1": "اسباب بازی", "cat2": None,
                  "sub_cat": "toy", "brand": "A"}]
            ).to_parquet(products_path, index=False)
            pd.DataFrame(
                [{"product_id": "1", "disadvantages": "قیمت بالا",
                  "recommendation_status": "not_recommended", "rate": 2.0}]
            ).to_parquet(comments_path, index=False)

            load_products.cache_clear()
            orchestrator = ShoppingAssistantOrchestrator(
                router=RuleBasedIntentRouter(),
                handlers={
                    "category_analytics": CategoryAnalyticsHandler(
                        products_path=products_path,
                        comments_path=comments_path,
                        client=None,
                    )
                },
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                result = orchestrator.run(
                    "پرتکرارترین شکایت کاربران در این دسته اسباب بازی چیست؟"
                )
            load_products.cache_clear()

        self.assertEqual(result.status, "success")
        self.assertEqual(result.route.intent, "category_analytics")
        self.assertIn("قیمت بالا", result.answer)
        self.assertEqual(result.payload["top_complaints"][0]["complaint"], "قیمت بالا")

    def test_registered_comparison_handler_receives_resolved_ids(self) -> None:
        handler = RecordingHandler()
        orchestrator = ShoppingAssistantOrchestrator(
            router=RuleBasedIntentRouter(),
            handlers={"product_comparison": handler},
        )
        result = orchestrator.run(
            "[product:10] و [product:20] را مقایسه کن"
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.payload["product_ids"], ["10", "20"])
        self.assertEqual(handler.requests[0].product_ids, ["10", "20"])

    def test_handler_failure_is_isolated_in_a_structured_result(self) -> None:
        orchestrator = ShoppingAssistantOrchestrator(
            router=RuleBasedIntentRouter(),
            handlers={"product_discovery": FailingHandler()},
        )
        result = orchestrator.run("یک کیف معرفی کن")
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "RuntimeError: boom")


if __name__ == "__main__":
    unittest.main()
