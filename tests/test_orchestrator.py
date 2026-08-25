"""Offline tests for Benyamin's chain orchestrator."""

from __future__ import annotations

import unittest

from src.orchestrator import (
    HandlerResult,
    OrchestratorRequest,
    RuleBasedIntentRouter,
    ShoppingAssistantOrchestrator,
    build_default_orchestrator,
    extract_product_ids,
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
        result = build_default_orchestrator().run(
            "ایرادهای پرتکرار این محصول چیست؟",
            context_product_ids=["3901234"],
        )
        self.assertEqual(result.status, "dependency_unavailable")
        self.assertEqual(result.route.product_ids, ["3901234"])
        self.assertNotIn("one_product_id", result.missing_requirements)

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

    def test_product_qa_waits_cleanly_for_comment_retriever(self) -> None:
        result = build_default_orchestrator().run(
            "ایرادهای پرتکرار این محصول چیست؟",
            product_ids=["3901234"],
        )
        self.assertEqual(result.status, "dependency_unavailable")
        self.assertEqual(result.route.intent, "product_qa")
        self.assertIn("comment_retriever", result.missing_requirements)

    def test_default_orchestrator_runs_product_discovery_end_to_end(self) -> None:
        result = build_default_orchestrator().run(
            "یک کیف روزمره زیر ۲۰۰ هزار تومان معرفی کن",
            top_k=3,
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.route.intent, "product_discovery")
        self.assertTrue(result.citations)
        self.assertIn("محصول‌های پیشنهادی", result.answer)

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
