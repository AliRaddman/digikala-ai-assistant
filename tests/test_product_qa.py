"""Offline tests for ProductQAChain (section 2: review-grounded QA)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.chains.product_qa import (
    NO_EVIDENCE_ANSWER_FA,
    ProductQAAnswer,
    ProductQAChain,
)
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

    def test_a_wholly_invented_claim_is_dropped_and_recorded(self) -> None:
        """Was assertRaises before 2026-08-30; see _quarantine_invented_ids."""
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
            result = chain.run("ایراد این محصول چیست؟", product_id="3901234")

        report = result.citation_hallucination
        self.assertEqual(result.answer.claims, [])
        self.assertEqual(report.invented_ids, ["00000000"])
        self.assertEqual(report.generated_ids, 1)
        self.assertEqual(report.rate, 1.0)
        self.assertEqual(len(report.dropped_claims), 1)
        self.assertEqual(report.dropped_claims[0]["text"], "ادعای ساختگی")
        # the invented id must never reach the rendered answer
        self.assertNotIn("00000000", result.render_fa())

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


class CommentIdFormatTests(unittest.TestCase):
    """The bracket bug from the live grounding run, guarded for this chain too.

    Added by Ali on 2026-08-30. SYSTEM_PROMPT asks for a bare comment_id, but
    the live judge showed a model will happily emit the full citation tag from
    an identically worded instruction. FakeProvider above always sends the
    bare form, which is exactly the blind spot that let 34 real judge calls
    fail while the offline suite stayed green.
    """

    def _run(self, comment_ids: list[str]) -> object:
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(
                response={
                    "answer_fa": "کاربران به زیپ ضعیف اشاره کرده‌اند.",
                    "sufficient_evidence": True,
                    "claims": [
                        {
                            "text": "زیپ بعد از مدتی خراب می‌شود.",
                            "comment_ids": comment_ids,
                        }
                    ],
                }
            )
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(provider, Path(directory)),
            )
            return chain.run("ایرادهای پرتکرار این محصول چیست؟", product_id="3901234")

    def test_a_valid_id_wrapped_in_a_citation_tag_is_accepted(self) -> None:
        result = self._run(["[comment:51230044]"])

        self.assertEqual(result.answer.claims[0].comment_ids, ["51230044"])
        # re-wrapped exactly once, so audit_citations' regex still matches
        self.assertIn("[comment:51230044]", result.render_fa())
        self.assertNotIn("[comment:[", result.render_fa())

    def test_a_valid_id_with_a_bare_prefix_is_accepted(self) -> None:
        result = self._run(["comment:51230044"])

        self.assertEqual(result.answer.claims[0].comment_ids, ["51230044"])

    def test_an_invented_id_inside_a_citation_tag_is_still_caught(self) -> None:
        result = self._run(["[comment:00000000]"])

        self.assertEqual(result.citation_hallucination.invented_ids, ["00000000"])
        self.assertEqual(result.answer.claims, [])


class AnswerSchemaContractTests(unittest.TestCase):
    """The schema the model is actually handed, not just the one we validate.

    Ali, 2026-08-30. The first live call answered "there are not enough
    reviews" over 20 reviews full of complaints, because `claims` carried a
    pydantic default: that kept it out of the schema's `required` list, so
    omitting it was legal, and it was declared after answer_fa, so the prose
    had to be generated before any evidence was enumerated. Every offline test
    passed throughout -- FakeProvider hands over claims no matter what the
    schema says.
    """

    def test_claims_are_required_and_generated_before_the_prose(self) -> None:
        schema = ProductQAAnswer.model_json_schema()

        self.assertIn("claims", schema["required"])
        self.assertEqual(
            list(schema["properties"]),
            ["claims", "sufficient_evidence", "answer_fa"],
        )

    def test_an_answer_without_claims_is_rejected_outright(self) -> None:
        with self.assertRaises(ValueError):
            ProductQAAnswer.model_validate(
                {"answer_fa": "متن نمونه", "sufficient_evidence": False}
            )


class CitationQuarantineTests(unittest.TestCase):
    """Partly-invented citations: keep the sound half, measure the rest.

    Ali, 2026-08-30. Modelled on the first live answer for product 262958,
    which cited 15 ids of which 12 were real and 3 were invented.
    """

    def _run(self, claims: list[dict]) -> object:
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(
                response={
                    "answer_fa": "خلاصه‌ی نظرات کاربران.",
                    "sufficient_evidence": True,
                    "claims": claims,
                }
            )
            chain = ProductQAChain(
                retriever=MockRetriever("comment"),
                client=_build_client(provider, Path(directory)),
            )
            return chain.run("ایرادهای پرتکرار این محصول چیست؟", product_id="3901234")

    def test_a_claim_keeps_its_real_ids_and_loses_only_the_invented_one(self) -> None:
        result = self._run(
            [{"text": "زیپ خراب می‌شود.", "comment_ids": ["51230044", "3612075"]}]
        )

        report = result.citation_hallucination
        self.assertEqual(result.answer.claims[0].comment_ids, ["51230044"])
        self.assertEqual(report.invented_ids, ["3612075"])
        self.assertEqual(report.generated_ids, 2)
        self.assertEqual(report.rate, 0.5)
        self.assertEqual(report.dropped_claims, [])

    def test_a_supported_claim_survives_beside_an_unsupported_one(self) -> None:
        result = self._run(
            [
                {"text": "زیپ خراب می‌شود.", "comment_ids": ["51230044"]},
                {"text": "ادعای بی‌پشتوانه.", "comment_ids": ["3612075", "3662052"]},
            ]
        )

        report = result.citation_hallucination
        self.assertEqual([c.text for c in result.answer.claims], ["زیپ خراب می‌شود."])
        self.assertEqual(report.invented_ids, ["3612075", "3662052"])
        self.assertEqual(len(report.dropped_claims), 1)

    def test_the_rendered_answer_warns_instead_of_looking_clean(self) -> None:
        rendered = self._run(
            [{"text": "زیپ خراب می‌شود.", "comment_ids": ["51230044", "3612075"]}]
        ).render_fa()

        self.assertIn("هشدار استناد", rendered)
        self.assertIn("1 از 2", rendered)
        self.assertNotIn("3612075", rendered)

    def test_a_clean_answer_carries_no_warning(self) -> None:
        result = self._run(
            [{"text": "زیپ خراب می‌شود.", "comment_ids": ["51230044"]}]
        )

        self.assertIsNone(result.hallucination_warning_fa())
        self.assertEqual(result.citation_hallucination.rate, 0.0)
        self.assertEqual(result.citation_hallucination.invented_ids, [])
        self.assertNotIn("هشدار", result.render_fa())


if __name__ == "__main__":
    unittest.main()
