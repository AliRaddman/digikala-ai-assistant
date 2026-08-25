"""Real product retrievers behind the interface in base.py.

Owner: Ali. Chains keep calling `build_retriever()` and never touch this
module directly, so switching from the mock is a one-line env change.

Indexes are loaded lazily and cached at module level: the FAISS index and the
BM25 matrix together are hundreds of megabytes, and a chain that constructs a
retriever per request must not pay that twice.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.normalize import tokenize
from src.retrieval.base import Evidence, RetrievalFilters, Retriever

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/indexes"))
MODEL_ID = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
IVF_NPROBE = 32
DEFAULT_INDEX_TYPE = os.getenv("INDEX_TYPE", "ivfsq8")
OVER_FETCH_STEPS = (10, 50, 200)

STOPWORDS = {
    "یه", "یک", "که", "را", "رو", "از", "به", "با", "برای", "در", "و", "تا",
    "این", "آن", "هم", "می", "خوام", "میخوام", "می‌خوام", "باشه", "باشد",
    "است", "بود", "شود", "کن", "کنید", "چند", "چه", "خیلی", "نباشه", "هست",
    "بهترین", "معرفی", "دنبال", "میگردم", "چیه", "داره", "دارم", "بده",
}


@lru_cache(maxsize=1)
def load_meta(index_dir: str = str(INDEX_DIR)) -> pd.DataFrame:
    df = pd.read_parquet(Path(index_dir) / "products_meta_v1.parquet")
    df["product_id"] = df["product_id"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_bm25(index_dir: str = str(INDEX_DIR)) -> tuple[sparse.csc_matrix, dict[str, int]]:
    path = Path(index_dir)
    matrix = sparse.load_npz(path / "products_bm25_v1.npz").tocsc()
    vocab = json.loads((path / "products_bm25_vocab_v1.json").read_text())
    return matrix, vocab


@lru_cache(maxsize=2)
def load_faiss(index_type: str = "ivfpq", index_dir: str = str(INDEX_DIR)):
    import faiss

    index = faiss.read_index(str(Path(index_dir) / f"products_e5base_{index_type}_v1.faiss"))
    if hasattr(index, "nprobe"):
        index.nprobe = IVF_NPROBE
    return index


@lru_cache(maxsize=1)
def load_encoder(model_id: str = MODEL_ID):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.max_seq_length = 128
    return model


def filter_mask(meta: pd.DataFrame, filters: RetrievalFilters | None) -> np.ndarray | None:
    """Boolean mask over the index rows.

    Applied as a hard constraint rather than a scoring bonus: when a user says
    "under 500 thousand", a cheaper-but-slightly-less-similar product is not a
    better answer, it is the only acceptable kind of answer.
    """
    if filters is None or filters.is_empty():
        return None
    mask = np.ones(len(meta), dtype=bool)

    if filters.price_min is not None:
        mask &= (meta["price"] >= filters.price_min).to_numpy(na_value=False)
    if filters.price_max is not None:
        mask &= (meta["price"] <= filters.price_max).to_numpy(na_value=False)
    if filters.brands:
        mask &= meta["brand"].isin(filters.brands).to_numpy()
    if filters.cat1:
        mask &= meta["cat1"].isin(filters.cat1).to_numpy()
    if filters.sub_cat:
        mask &= meta["sub_cat"].isin(filters.sub_cat).to_numpy()
    if filters.min_rate is not None:
        mask &= (meta["rate"] >= filters.min_rate).to_numpy(na_value=False)
    if filters.min_rate_count is not None:
        mask &= (meta["rate_count"] >= filters.min_rate_count).to_numpy()
    if filters.exclude_fake:
        mask &= ~meta["is_fake"].to_numpy()
    if filters.product_ids:
        mask &= meta["product_id"].isin(filters.product_ids).to_numpy()
    return mask


def to_evidence(meta: pd.DataFrame, positions: np.ndarray, scores: np.ndarray) -> list[Evidence]:
    rows = meta.iloc[positions]
    return [
        Evidence(
            id=row.product_id,
            kind="product",
            text=row.title,
            score=float(score),
            product_id=row.product_id,
            title=row.title,
            meta={
                "brand": row.brand,
                "price": None if pd.isna(row.price) else float(row.price),
                "rate": None if pd.isna(row.rate) else float(row.rate),
                "rate_count": int(row.rate_count),
                "cat1": row.cat1,
                "sub_cat": row.sub_cat,
                "is_fake": bool(row.is_fake),
            },
        )
        for row, score in zip(rows.itertuples(index=False), scores)
    ]


def top_positions(
    scores: np.ndarray, mask: np.ndarray | None, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Highest-scoring allowed rows, sorted descending.

    A None mask means no filter was requested. That case skips building and
    applying a 948k-element boolean array, which was costing about 10 ms per
    call on the unfiltered path — more than the query encoding itself.
    """
    if mask is None:
        limit = min(top_k, len(scores))
    else:
        scores = np.where(mask, scores, -np.inf)
        limit = min(top_k, int(mask.sum()))
    if limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    order = candidates[np.argsort(-scores[candidates])]
    return order, scores[order]


class BM25Retriever(Retriever):
    """Lexical retrieval over the pre-weighted sparse matrix.

    Scoring touches every document, which sounds expensive but is a handful of
    sparse column sums, and it means filters are exact: there is no candidate
    list to run out of.
    """

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index_dir = str(index_dir)

    def retrieve(
        self, query: str, top_k: int = 10, filters: RetrievalFilters | None = None
    ) -> list[Evidence]:
        matrix, vocab = load_bm25(self.index_dir)
        meta = load_meta(self.index_dir)

        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        term_ids = [vocab[t] for t in terms or tokenize(query) if t in vocab]
        if not term_ids:
            return []

        scores = np.asarray(matrix[:, term_ids].sum(axis=1)).ravel()
        positions, values = top_positions(scores, filter_mask(meta, filters), top_k)
        return to_evidence(meta, positions, values)


class DenseRetriever(Retriever):
    """Semantic retrieval over the FAISS index.

    FAISS cannot express the structured filters, so the query is over-fetched
    and filtered afterwards. When a narrow filter empties the first batch the
    fetch widens instead of silently returning fewer results; the fallback is
    bounded so a filter matching almost nothing fails fast rather than
    scanning the whole catalogue.
    """

    def __init__(self, index_type: str = DEFAULT_INDEX_TYPE, index_dir: Path = INDEX_DIR) -> None:
        self.index_type = index_type
        self.index_dir = str(index_dir)

    def encode_query(self, query: str) -> np.ndarray:
        model = load_encoder()
        vector = model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        )
        return vector.astype("float32")

    def retrieve(
        self, query: str, top_k: int = 10, filters: RetrievalFilters | None = None
    ) -> list[Evidence]:
        index = load_faiss(self.index_type, self.index_dir)
        meta = load_meta(self.index_dir)
        mask = filter_mask(meta, filters)
        vector = self.encode_query(query)

        for step in OVER_FETCH_STEPS:
            depth = min(top_k * step, index.ntotal)
            distances, positions = index.search(vector, depth)
            scores = 1.0 - distances / 2.0
            positions, scores = positions[0], scores[0]
            keep = positions >= 0
            if mask is not None:
                keep &= mask[positions]
            if keep.sum() >= top_k or depth >= index.ntotal:
                positions, scores = positions[keep][:top_k], scores[keep][:top_k]
                return to_evidence(meta, positions, scores)
        return []


def build_product_retriever(mode: str | None = None) -> Retriever:
    """Factory used by base.build_retriever when RETRIEVER_MODE is not mock."""
    backend = (mode or os.getenv("RETRIEVER_BACKEND", "dense")).lower()
    if backend == "hybrid":
        from src.retrieval.hybrid import HybridRetriever
        return HybridRetriever()
    if backend == "dense":
        return DenseRetriever()
    if backend == "bm25":
        return BM25Retriever()
    raise ValueError(f"unknown RETRIEVER_BACKEND: {backend!r}")


if __name__ == "__main__":
    query = "یه کوله پشتی مناسب مدرسه می‌خوام که جادار باشه"
    for name, retriever in (("bm25", BM25Retriever()), ("dense", DenseRetriever())):
        print(f"--- {name}")
        for ev in retriever.retrieve(query, top_k=5):
            print(f"  {ev.score:.3f} {ev.citation()} {ev.title}")

    print("--- dense + filter price < 1,000,000 rial")
    filters = RetrievalFilters(price_max=1_000_000, exclude_fake=True)
    for ev in DenseRetriever().retrieve(query, top_k=5, filters=filters):
        print(f"  {ev.score:.3f} {ev.citation()} {ev.title} | {ev.meta['price']}")