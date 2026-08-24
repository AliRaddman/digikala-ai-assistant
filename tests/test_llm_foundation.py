"""Offline tests for Benyamin's LLM foundation work."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.chains.product_discovery import ProductDiscoveryChain
from src.chains.product_filters import ProductFilterPlan, RuleBasedFilterExtractor
from src.llm.cache import SQLiteLLMCache
from src.llm.client import CachedLLMClient, ProviderResult
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
            self.assertEqual(provider.calls, 1)
            self.assertEqual(second.data, {"value": "سلام"})
            summary = ledger.summary()
            self.assertEqual(summary["logical_requests"], 2)
            self.assertEqual(summary["api_calls"], 1)
            self.assertEqual(summary["cache_hits"], 1)
            self.assertGreater(summary["saved_cost_usd"], 0)

    def test_toman_price_is_converted_to_rial(self) -> None:
        plan = RuleBasedFilterExtractor().extract(
            "شلوار جین مردانه راحت زیر ۵۰۰ هزار تومن"
        )
        self.assertEqual(plan.price_max_rial, 5_000_000)
        self.assertEqual(plan.search_query, "شلوار جین مردانه راحت")

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
