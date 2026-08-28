"""Real comment retriever behind the interface in base.py.

Owner: Ali. Chains keep calling `build_retriever()` and never touch this
module directly.

Unlike products, comment retrieval is always scoped to a product (see
docs/DECISIONS.md, "لایه نظرات"): the index is a flat, mmap-read embedding
array plus a product_id -> row-indices map, so a scoped query is one exact
dot product over a few dozen candidate rows rather than an approximate
nearest-neighbour search. This module does not use the comments BM25 index
built alongside it (comments_bm25_v1.npz) -- that artefact exists for
completeness and possible future use, but the design here is dense-only.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.base import Evidence, RetrievalFilters, Retriever

INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/indexes"))
MODEL_ID = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "


@lru_cache(maxsize=1)
def load_meta(index_dir: str = str(INDEX_DIR)) -> pd.DataFrame:
    df = pd.read_parquet(Path(index_dir) / "comments_meta_v1.parquet")
    df["comment_id"] = df["comment_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_embeddings(index_dir: str = str(INDEX_DIR)) -> np.ndarray:
    return np.load(Path(index_dir) / "comments_emb_e5base_v1.npy", mmap_mode="r")


@lru_cache(maxsize=1)
def load_product_map(index_dir: str = str(INDEX_DIR)) -> dict[str, list[int]]:
    path = Path(index_dir) / "comments_product_map_v1.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_encoder(model_id: str = MODEL_ID):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.max_seq_length = 128
    return model


def _text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _to_evidence(row: pd.Series, score: float) -> Evidence:
    title = _text_or_none(row.title)
    return Evidence(
        id=row.comment_id,
        kind="comment",
        text=row.body,
        score=score,
        product_id=row.product_id,
        title=title,
        meta={
            "rate": None if pd.isna(row.rate) else float(row.rate),
            "recommendation_status": (
                None if pd.isna(row.recommendation_status) else row.recommendation_status
            ),
            "is_buyer": bool(row.is_buyer),
            "likes": int(row.likes),
            "advantages": _text_or_none(row.advantages),
            "disadvantages": _text_or_none(row.disadvantages),
            "title": title,
        },
    )


class CommentRetriever(Retriever):
    """Dense retrieval over the per-product comment index.

    `filters.product_ids` is this retriever's primary filter: it narrows the
    dot product to the handful of rows that belong to those products. With no
    product filter, the query is scored against every row in the index --
    correct, but a full 3-4M-row scan, so it exists as a fallback rather than
    the intended usage.
    """

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index_dir = str(index_dir)

    def encode_query(self, query: str) -> np.ndarray:
        model = load_encoder()
        vector = model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        )
        return vector.astype("float32")[0]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:
        meta = load_meta(self.index_dir)
        embeddings = load_embeddings(self.index_dir)

        if filters is not None and filters.product_ids:
            product_map = load_product_map(self.index_dir)
            positions = np.array(
                [
                    position
                    for product_id in filters.product_ids
                    for position in product_map.get(product_id, [])
                ],
                dtype=np.int64,
            )
            if positions.size == 0:
                return []
            query_vector = self.encode_query(query)
            scores = embeddings[positions] @ query_vector
        else:
            query_vector = self.encode_query(query)
            scores = np.asarray(embeddings) @ query_vector
            positions = np.arange(len(meta))

        limit = min(top_k, len(positions))
        if limit <= 0:
            return []
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        chosen_positions = positions[top]
        chosen_scores = scores[top]

        return [
            _to_evidence(meta.iloc[position], float(score))
            for position, score in zip(chosen_positions, chosen_scores)
        ]


if __name__ == "__main__":
    retriever = CommentRetriever()
    sample_product_id = next(iter(load_product_map(str(INDEX_DIR))))
    filters = RetrievalFilters(product_ids=[sample_product_id])

    print(f"--- comments for product {sample_product_id}, filtered")
    for ev in retriever.retrieve("ایراد این محصول چیست؟", top_k=5, filters=filters):
        print(f"  {ev.score:.3f} {ev.citation()} {ev.text[:80]}")

    print("--- same query, unfiltered (full-index scan)")
    for ev in retriever.retrieve("ایراد این محصول چیست؟", top_k=3):
        print(f"  {ev.score:.3f} {ev.citation()} {ev.product_id} {ev.text[:80]}")
