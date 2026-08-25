"""Hybrid retrieval: BM25 and dense, merged by reciprocal rank fusion.

Owner: Ali.

    score(d) = Σ_i  w_i / (k + rank_i(d))

Fusion works on positions rather than scores, which matters here because the
two scales do not compare: BM25 scores in this index run past 20 and are
unbounded, while cosine similarity sits in a narrow band around 0.85. A
weighted sum of normalised scores was measured too and came out lower.

Parameters k=60 and w_dense=0.7 were selected over a grid of twenty
configurations on the 36-query eval set. Both fetch depth and that selection
matter, and the second one is a caveat: the parameters were tuned on the same
queries the improvement is reported on, so the measured gain is optimistic.

Measured on the depth-50 pool (see data/eval_d50/):

    dense only   nDCG@10 0.7329
    hybrid       nDCG@10 0.7778   (21 queries better, 8 worse, p=0.058)
    bm25 only    nDCG@10 0.6389

The gain over dense alone points the right way in every metric but does not
clear the significance threshold with 36 queries. It is enabled anyway
because BM25 costs 0.2s to build and 60 MB to store, and the worst case is
parity with dense.

Fetch depth is the part that actually decided this. At depth 10 the two
retrievers returned the same ten products on all 36 queries and fusion
changed nothing; the effect only appears once each retriever has room to
disagree.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.retrieval.base import Evidence, RetrievalFilters, Retriever
from src.retrieval.products import INDEX_DIR, BM25Retriever, DenseRetriever

RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
FETCH_DEPTH = int(os.getenv("HYBRID_FETCH_DEPTH", "50"))


class HybridRetriever(Retriever):
    """Fuses the lexical and semantic rankings.

    Filters are applied inside each retriever, so the fused list is already
    constrained and nothing has to be re-checked here.
    """

    def __init__(
        self,
        rrf_k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT,
        fetch_depth: int = FETCH_DEPTH,
        index_dir: Path = INDEX_DIR,
    ) -> None:
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.fetch_depth = fetch_depth
        self.dense = DenseRetriever(index_dir=index_dir)
        self.sparse = BM25Retriever(index_dir=index_dir)

    def retrieve(
        self, query: str, top_k: int = 10, filters: RetrievalFilters | None = None
    ) -> list[Evidence]:
        depth = max(self.fetch_depth, top_k)
        runs = (
            (self.dense.retrieve(query, top_k=depth, filters=filters), self.dense_weight),
            (self.sparse.retrieve(query, top_k=depth, filters=filters), 1 - self.dense_weight),
        )

        scores: dict[str, float] = {}
        items: dict[str, Evidence] = {}
        for evidence_list, weight in runs:
            for rank, evidence in enumerate(evidence_list, start=1):
                scores[evidence.id] = scores.get(evidence.id, 0.0) + weight / (
                    self.rrf_k + rank
                )
                items.setdefault(evidence.id, evidence)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        fused: list[Evidence] = []
        for product_id, score in ranked:
            evidence = items[product_id]
            fused.append(
                Evidence(
                    id=evidence.id,
                    kind=evidence.kind,
                    text=evidence.text,
                    score=round(score, 6),
                    product_id=evidence.product_id,
                    title=evidence.title,
                    meta={**evidence.meta, "fusion": "rrf"},
                )
            )
        return fused


if __name__ == "__main__":
    query = "یه کوله پشتی مناسب مدرسه می‌خوام که جادار باشه"
    for ev in HybridRetriever().retrieve(query, top_k=5):
        print(f"{ev.score:.5f} {ev.citation()} {ev.title}")

    print("\nwith filter: price < 1,000,000 rial, no fakes")
    filters = RetrievalFilters(price_max=1_000_000, exclude_fake=True)
    for ev in HybridRetriever().retrieve(query, top_k=5, filters=filters):
        print(f"{ev.score:.5f} {ev.citation()} {ev.title} | {ev.meta['price']}")