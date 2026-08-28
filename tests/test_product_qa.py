"""Offline tests for ProductQAChain (section 2: review-grounded QA)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.chains.product_qa import NO_EVIDENCE_ANSWER_FA, ProductQAChain
from src.llm.cache import SQLiteLLMCache
from src.llm.client import CachedLLMClient, ProviderResult
from src.llm.types import TokenUsage
from src.llm.usage import SQLiteUsageLedger
from src.retrieval.base import MockRetriever


class FakeProvider:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = 0

    def generate_structured(self, *, model, messages, response_model):
        self.calls += 1
        return ProviderResult(
            data=self.response,
            model=model,
            request_id="resp_test",
            usage=TokenUsage(input_tokens=200, output_tokens=40),
        )


def _build_client(provider: FakeProvider, root: Path) -> CachedLLMClient:
    return CachedLLMClient(
        provider=provider,
        model="gpt-4o-mini",
        cache=SQLiteLLMCache(root / "cache.sqlite3"),
        ledger=SQLiteUsageLedger(root / "usage.sqlite3"),
    )


class ProductQAChainTests(unittest.TestCase):
    def test_no_reviews_short_circuits_without_calling_the_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(response={})
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(provider, Path(directory)),
            )
            result = chain.run("ایراد این محصول چیست؟", product_id="9999999")

        self.assertEqual(result.answer.answer_fa, NO_EVIDENCE_ANSWER_FA)
        self.assertFalse(result.answer.sufficient_evidence)
        self.assertEqual(result.evidence, [])
        self.assertEqual(provider.calls, 0)

    def test_grounded_answer_cites_a_real_comment_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(
                response={
                    "answer_fa": "کاربران به زیپ ضعیف اشاره کرده‌اند.",
                    "sufficient_evidence": True,
                    "claims": [
                        {
                            "text": "زیپ بعد از مدتی خراب می‌شود.",
                            "comment_ids": ["51230044"],
                        }
                    ],
                }
            )
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(provider, Path(directory)),
            )
            result = chain.run("ایرادهای پرتکرار این محصول چیست؟", product_id="3901234")

        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.evidence)
        self.assertIn("[comment:51230044]", result.render_fa())

    def test_insufficient_evidence_is_a_valid_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(
                response={
                    "answer_fa": "نظرات کافی برای پاسخ به این سؤال وجود ندارد.",
                    "sufficient_evidence": False,
                    "claims": [],
                }
            )
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(provider, Path(directory)),
            )
            result = chain.run("آیا اندازه‌اش دقیق است؟", product_id="3901234")

        self.assertFalse(result.answer.sufficient_evidence)
        self.assertEqual(result.answer.claims, [])
        self.assertEqual(result.render_fa(), "نظرات کافی برای پاسخ به این سؤال وجود ندارد.")

    def test_a_hallucinated_comment_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(
                response={
                    "answer_fa": "متن نمونه",
                    "sufficient_evidence": True,
                    "claims": [
                        {"text": "ادعای ساختگی", "comment_ids": ["00000000"]}
                    ],
                }
            )
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(provider, Path(directory)),
            )
            with self.assertRaises(ValueError):
                chain.run("ایراد این محصول چیست؟", product_id="3901234")

    def test_empty_question_or_product_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(FakeProvider(response={}), Path(directory)),
            )
            with self.assertRaises(ValueError):
                chain.run("  ", product_id="3901234")
            with self.assertRaises(ValueError):
                chain.run("ایراد چیست؟", product_id=" ")


if __name__ == "__main__":
    unittest.main()
