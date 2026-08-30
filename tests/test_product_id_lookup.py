"""Offline tests for addressing products by id through the shared Retriever.

Owner: Ali, 2026-08-30. `RetrievalFilters.product_ids` is documented in
base.py as a hard filter, but on the product side neither backend could
reliably honour it: DenseRetriever post-filters an over-fetched FAISS
candidate list, and BM25Retriever returns nothing when every query term is a
stopword. Both failed silently, which is how Fatemeh's comparison chain came
to report "شناسه‌های محصول پیدا نشد" for two products that are in the index.

These tests run against a small synthetic metadata frame, so they need no
index on disk and cover the lookup itself rather than either search backend.
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.retrieval.base import RetrievalFilters
from src.retrieval.products import exact_product_evidence


def _meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["4835951", "3843544", "1120876"],
            "title": ["میکروفن کندانسر", "کفش زنانه مانگو", "کوله پشتی"],
            "brand": ["متفرقه", "مانگو", "متفرقه"],
            "price": [12900000.0, 13462000.0, 990000.0],
            "rate": [74.0, 100.0, float("nan")],
            "rate_count": [245, 1, 0],
            "cat1": ["کالای دیجیتال", "کفش زنانه", "کیف"],
            "sub_cat": ["میکروفن", "کفش", "کوله"],
            "is_fake": [False, False, True],
        }
    )


class ExactProductLookupTests(unittest.TestCase):
    def test_requested_products_are_returned_in_the_order_asked_for(self) -> None:
        evidence = exact_product_evidence(
            _meta(),
            RetrievalFilters(product_ids=["3843544", "4835951"]),
            top_k=10,
        )

        self.assertEqual([item.id for item in evidence], ["3843544", "4835951"])
        self.assertEqual(evidence[0].kind, "product")
        self.assertEqual(evidence[0].citation(), "[product:3843544]")
        self.assertEqual(evidence[1].title, "میکروفن کندانسر")
        self.assertEqual(evidence[1].meta["price"], 12900000.0)

    def test_an_unknown_id_is_skipped_rather_than_faked(self) -> None:
        evidence = exact_product_evidence(
            _meta(),
            RetrievalFilters(product_ids=["4835951", "0000000"]),
            top_k=10,
        )

        self.assertEqual([item.id for item in evidence], ["4835951"])

    def test_score_is_zero_because_nothing_was_ranked(self) -> None:
        evidence = exact_product_evidence(
            _meta(), RetrievalFilters(product_ids=["4835951"]), top_k=10
        )

        self.assertEqual(evidence[0].score, 0.0)

    def test_other_filters_still_apply_alongside_the_ids(self) -> None:
        """An id filter addresses rows; it does not switch the others off."""
        evidence = exact_product_evidence(
            _meta(),
            RetrievalFilters(product_ids=["1120876", "4835951"], exclude_fake=True),
            top_k=10,
        )

        self.assertEqual([item.id for item in evidence], ["4835951"])

    def test_top_k_bounds_the_result(self) -> None:
        evidence = exact_product_evidence(
            _meta(),
            RetrievalFilters(product_ids=["4835951", "3843544", "1120876"]),
            top_k=2,
        )

        self.assertEqual(len(evidence), 2)


if __name__ == "__main__":
    unittest.main()
