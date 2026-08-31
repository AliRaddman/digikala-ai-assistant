"""Offline tests for the RRF fusion behind the headline retrieval number.

Owner: Ali, 2026-08-30. src/retrieval/hybrid.py had no test at all, while
nDCG@10 = 0.7778 -- the number the README leads with, and the configuration
RETRIEVER_BACKEND now selects -- comes out of it. These stub the two component
retrievers so the fusion arithmetic is checked without any index on disk.
"""

from __future__ import annotations

import unittest

from src.retrieval.base import Evidence, RetrievalFilters, Retriever
from src.retrieval.hybrid import HybridRetriever


def _evidence(identifier: str) -> Evidence:
    return Evidence(
        id=identifier,
        kind="product",
        text=f"محصول {identifier}",
        score=1.0,
        product_id=identifier,
        title=f"محصول {identifier}",
        meta={},
    )


class StubRetriever(Retriever):
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[int, RetrievalFilters | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:
        self.calls.append((top_k, filters))
        return [_evidence(identifier) for identifier in self.ids[:top_k]]


class HybridFusionTests(unittest.TestCase):
    def _retriever(self, dense: list[str], sparse: list[str]) -> HybridRetriever:
        retriever = HybridRetriever()
        retriever.dense = StubRetriever(dense)
        retriever.sparse = StubRetriever(sparse)
        return retriever

    def test_defaults_match_the_evaluated_configuration(self) -> None:
        """data/eval/hybrid_d50_v2.csv's best row is k=60, w_dense=0.7 at depth 50.

        If these drift, the README's 0.7778 stops describing what runs.
        """
        retriever = HybridRetriever()

        self.assertEqual(retriever.rrf_k, 60)
        self.assertEqual(retriever.dense_weight, 0.7)
        self.assertEqual(retriever.fetch_depth, 50)

    def test_a_document_both_rankers_return_outranks_one_only_dense_has(self) -> None:
        fused = self._retriever(dense=["a", "b"], sparse=["b", "c"]).retrieve("q", top_k=3)

        self.assertEqual([item.id for item in fused], ["b", "a", "c"])

    def test_dense_outweighs_sparse_at_equal_rank(self) -> None:
        fused = self._retriever(dense=["a"], sparse=["b"]).retrieve("q", top_k=2)

        self.assertEqual([item.id for item in fused], ["a", "b"])
        self.assertGreater(fused[0].score, fused[1].score)

    def test_fusion_is_marked_in_the_metadata(self) -> None:
        fused = self._retriever(dense=["a"], sparse=["a"]).retrieve("q", top_k=1)

        self.assertEqual(fused[0].meta["fusion"], "rrf")
        self.assertEqual(len(fused), 1, "the same document must not appear twice")

    def test_both_rankers_are_over_fetched_and_filters_reach_them(self) -> None:
        """Filters are applied inside each retriever, never re-checked after."""
        retriever = self._retriever(dense=["a"], sparse=["a"])
        filters = RetrievalFilters(product_ids=["a"])

        retriever.retrieve("q", top_k=5, filters=filters)

        for stub in (retriever.dense, retriever.sparse):
            depth, passed = stub.calls[0]
            self.assertEqual(depth, retriever.fetch_depth)
            self.assertIs(passed, filters)

    def test_top_k_larger_than_the_fetch_depth_widens_the_fetch(self) -> None:
        retriever = self._retriever(dense=["a"], sparse=["a"])

        retriever.retrieve("q", top_k=200)

        self.assertEqual(retriever.dense.calls[0][0], 200)


if __name__ == "__main__":
    unittest.main()
