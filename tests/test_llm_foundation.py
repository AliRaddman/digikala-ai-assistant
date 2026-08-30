"""Offline tests for Benyamin's LLM foundation work."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.chains.product_discovery import ProductDiscoveryChain
from src.chains.product_filters import ProductFilterPlan, RuleBasedFilterExtractor
from src.llm.cache import SQLiteLLMCache
from src.llm.client import CachedLLMClient, ProviderResult
from src.llm.semantic_cache import SemanticCacheRequest
from src.llm.types import TokenUsage
from src.llm.usage import SQLiteUsageLedger, estimate_cost_usd
from src.retrieval.base import MockRetriever


class DemoSchema(BaseModel):
    value: str


class FakeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate_structured(self, *, model, messages, response_model):
        self.calls += 1
        return ProviderResult(
            data={"value": messages[-1]["content"]},
            model=model,
            request_id="resp_test",
            usage=TokenUsage(input_tokens=1_000, output_tokens=100),
        )


class FakeSemanticEncoder:
    model_id = "fake-semantic-v1"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def encode(self, text: str) -> list[float]:
        self.calls += 1
        return self.vectors[text]


class LLMFoundationTests(unittest.TestCase):
    def test_gpt_4o_mini_cost(self) -> None:
        usage = TokenUsage(
            input_tokens=1_000_000,
            cached_input_tokens=200_000,
            output_tokens=100_000,
        )
        self.assertAlmostEqual(estimate_cost_usd("gpt-4o-mini", usage), 0.195)

    def test_second_identical_request_uses_disk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider()
            ledger = SQLiteUsageLedger(root / "usage.sqlite3")
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=ledger,
            )
            kwargs = {
                "operation": "test",
                "messages": [{"role": "user", "content": "سلام"}],
                "response_model": DemoSchema,
                "cache_namespace": "test-v1",
            }
            first = client.generate_structured(**kwargs)
            second = client.generate_structured(**kwargs)

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(second.cache_type, "exact")
            self.assertEqual(provider.calls, 1)
            self.assertEqual(second.data, {"value": "سلام"})
            summary = ledger.summary()
            self.assertEqual(summary["logical_requests"], 2)
            self.assertEqual(summary["api_calls"], 1)
            self.assertEqual(summary["cache_hits"], 1)
            self.assertGreater(summary["saved_cost_usd"], 0)

    def test_semantic_paraphrase_reuses_a_guarded_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider()
            encoder = FakeSemanticEncoder(
                {
                    "کیف ارزان": [1.0, 0.0],
                    "یک کیف کم قیمت": [1.0, 0.0],
                }
            )
            ledger = SQLiteUsageLedger(root / "usage.sqlite3")
            cache = SQLiteLLMCache(root / "cache.sqlite3")
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=cache,
                ledger=ledger,
                semantic_encoder=encoder,
                semantic_threshold=0.95,
            )
            guard = {"system_prompt": "extract-v1", "catalogue": "same"}
            first = client.generate_structured(
                operation="test",
                messages=[{"role": "user", "content": "کیف ارزان"}],
                response_model=DemoSchema,
                cache_namespace="test-v1",
                semantic_cache=SemanticCacheRequest(
                    text="کیف ارزان", guard=guard
                ),
            )
            second = client.generate_structured(
                operation="test",
                messages=[{"role": "user", "content": "یک کیف کم قیمت"}],
                response_model=DemoSchema,
                cache_namespace="test-v1",
                semantic_cache=SemanticCacheRequest(
                    text="یک کیف کم قیمت", guard=guard
                ),
            )

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(second.cache_type, "semantic")
            self.assertAlmostEqual(second.cache_similarity or 0.0, 1.0, places=6)
            self.assertEqual(second.data, {"value": "کیف ارزان"})
            self.assertEqual(provider.calls, 1)
            self.assertEqual(encoder.calls, 2)
            summary = ledger.summary()
            self.assertEqual(summary["semantic_cache_hits"], 1)
            self.assertEqual(summary["exact_cache_hits"], 0)
            self.assertGreater(summary["saved_cost_usd"], 0)

    def test_semantic_guard_prevents_cross_context_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider()
            encoder = FakeSemanticEncoder({"سؤال یک": [1.0], "پرسش یک": [1.0]})
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
                semantic_encoder=encoder,
                semantic_threshold=0.95,
            )
            for text, evidence_id in [("سؤال یک", "comment:1"), ("پرسش یک", "comment:2")]:
                result = client.generate_structured(
                    operation="test",
                    messages=[{"role": "user", "content": text}],
                    response_model=DemoSchema,
                    cache_namespace="test-v1",
                    semantic_cache=SemanticCacheRequest(
                        text=text,
                        guard={"evidence_ids": [evidence_id]},
                    ),
                )
                self.assertFalse(result.cache_hit)

            self.assertEqual(provider.calls, 2)

    def test_semantic_similarity_must_clear_the_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider()
            encoder = FakeSemanticEncoder(
                {"کیف ارزان": [1.0, 0.0], "کیف گران": [0.8, 0.6]}
            )
            client = CachedLLMClient(
                provider=provider,
                model="gpt-4o-mini",
                cache=SQLiteLLMCache(root / "cache.sqlite3"),
                ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
                semantic_encoder=encoder,
                semantic_threshold=0.95,
            )
            for text in ["کیف ارزان", "کیف گران"]:
                result = client.generate_structured(
                    operation="test",
                    messages=[{"role": "user", "content": text}],
                    response_model=DemoSchema,
                    cache_namespace="test-v1",
                    semantic_cache=SemanticCacheRequest(text=text, guard={}),
                )
                self.assertFalse(result.cache_hit)

            self.assertEqual(provider.calls, 2)

    def test_usage_ledger_migrates_existing_exact_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE llm_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        model TEXT NOT NULL,
                        request_id TEXT,
                        cache_hit INTEGER NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        cached_input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        latency_ms REAL NOT NULL,
                        cost_usd REAL NOT NULL,
                        saved_cost_usd REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO llm_usage VALUES (
                        1, '2026-08-30T00:00:00+00:00', 'test',
                        'gpt-4o-mini', NULL, 1, 0, 0, 0, 0.2, 0.0, 0.001
                    )
                    """
                )

            summary = SQLiteUsageLedger(path).summary()

            self.assertEqual(summary["cache_hits"], 1)
            self.assertEqual(summary["exact_cache_hits"], 1)
            self.assertEqual(summary["semantic_cache_hits"], 0)

    def test_toman_price_is_converted_to_rial(self) -> None:
        plan = RuleBasedFilterExtractor().extract(
            "شلوار جین مردانه راحت زیر ۵۰۰ هزار تومن"
        )
        self.assertEqual(plan.price_max_rial, 5_000_000)
        self.assertEqual(plan.search_query, "شلوار جین مردانه راحت")

    def test_worded_toman_price_is_converted_to_rial(self) -> None:
        plan = RuleBasedFilterExtractor().extract(
            "عطر مردانه تلخ و ماندگار زیر یک میلیون تومن"
        )
        self.assertEqual(plan.price_max_rial, 10_000_000)
        self.assertEqual(plan.search_query, "عطر مردانه تلخ و ماندگار")

    def test_compound_worded_price_and_minimum_are_supported(self) -> None:
        plan = RuleBasedFilterExtractor().extract(
            "گوشی بیشتر از یک و نیم میلیون تومان"
        )
        self.assertEqual(plan.price_min_rial, 15_000_000)
        self.assertEqual(plan.search_query, "گوشی")

    def test_grouped_numeric_price_is_not_read_as_a_decimal(self) -> None:
        plan = RuleBasedFilterExtractor().extract(
            "کفش حداکثر 2,500,000 ریال"
        )
        self.assertEqual(plan.price_max_rial, 2_500_000)
        self.assertEqual(plan.search_query, "کفش")

    def test_unknown_model_does_not_silently_report_zero_cost(self) -> None:
        with self.assertRaises(ValueError):
            estimate_cost_usd("unknown-model", TokenUsage(input_tokens=100))

    def test_filter_plan_rejects_inverted_price_range(self) -> None:
        with self.assertRaises(ValueError):
            ProductFilterPlan(
                search_query="کیف",
                price_min_rial=2_000_000,
                price_max_rial=1_000_000,
            )

    def test_product_discovery_runs_end_to_end_offline(self) -> None:
        chain = ProductDiscoveryChain(
            retriever=MockRetriever("product"),
            extractor=RuleBasedFilterExtractor(),
        )
        result = chain.run("کیف روزمره زیر ۲۰۰ هزار تومان", top_k=3)

        self.assertEqual(result.filter_plan.price_max_rial, 2_000_000)
        self.assertTrue(result.products)
        self.assertTrue(
            all(product.meta["price"] <= 2_000_000 for product in result.products)
        )
        self.assertIn("[product:", result.render_fa())


if __name__ == "__main__":
    unittest.main()
