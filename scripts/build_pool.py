"""Run the candidate embedding models and BM25 over the eval queries, then
build the pool that gets labelled by hand.

Owner: Ali.

Nothing here judges relevance. It only produces, for every query, the union of
what each system would have shown, so that a single hand-labelled pool can
score all four systems fairly. Labelling only the winner's results would bias
the comparison towards whichever system was labelled first.

    python -m scripts.build_pool \
        --sample data/processed/products_sample_50k_v1.parquet \
        --queries data/eval/queries_v1.jsonl \
        --out-dir data/eval
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A candidate encoder.

    The e5 family was trained with explicit query/passage prefixes and loses
    noticeable quality without them, so the prefix belongs to the model, not
    to the caller.
    """

    name: str
    hf_id: str
    query_prefix: str = ""
    passage_prefix: str = ""


MODELS: list[ModelSpec] = [
    ModelSpec("e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: "),
    ModelSpec("bge-m3", "BAAI/bge-m3"),
    ModelSpec("e5-large", "intfloat/multilingual-e5-large", "query: ", "passage: "),
    ]

MAX_SEQ_LEN = 128
BATCH_SIZE = 128
POOL_DEPTH = 10


def build_dense_text(row: pd.Series) -> str:
    """Readable text for the encoder.

    Deliberately different from search_text: dense models are trained on
    fluent sentences, so this keeps the original casing and punctuation and
    drops the placeholder brand "متفرقه", which means "no brand" and appears
    on a large share of the catalogue.
    """
    parts = [row["title"]]
    brand = row["brand"]
    if isinstance(brand, str) and brand and brand != "متفرقه":
        parts.append(brand)
    for col in ("cat2", "cat1"):
        value = row[col]
        if isinstance(value, str) and value:
            parts.append(value)
            break
    return " | ".join(parts)


def load_queries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_bm25(
    docs: list[str], queries: list[dict], product_ids: list[str], depth: int
) -> tuple[list[dict], float]:
    from rank_bm25 import BM25Okapi

    from src.data.normalize import tokenize

    start = time.perf_counter()
    index = BM25Okapi([doc.split() for doc in docs])
    index_time = time.perf_counter() - start

    results: list[dict] = []
    for query in queries:
        scores = index.get_scores(tokenize(query["query"]))
        top = np.argsort(-scores)[:depth]
        for rank, idx in enumerate(top, start=1):
            results.append(
                {
                    "query_id": query["query_id"],
                    "product_id": product_ids[idx],
                    "rank": rank,
                    "score": float(scores[idx]),
                }
            )
    return results, index_time


def run_dense(
    spec: ModelSpec,
    texts: list[str],
    queries: list[dict],
    product_ids: list[str],
    depth: int,
    cache_dir: Path,
) -> tuple[list[dict], float]:
    import torch
    from sentence_transformers import SentenceTransformer

    cache_path = cache_dir / f"{spec.name}.npy"
    model = SentenceTransformer(
        spec.hf_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.max_seq_length = MAX_SEQ_LEN

    start = time.perf_counter()
    if cache_path.exists():
        doc_emb = np.load(cache_path)
        index_time = 0.0
    else:
        doc_emb = model.encode(
            [spec.passage_prefix + text for text in texts],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        index_time = time.perf_counter() - start
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, doc_emb)

    query_emb = model.encode(
        [spec.query_prefix + q["query"] for q in queries],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    sims = query_emb @ doc_emb.T
    results: list[dict] = []
    for row, query in enumerate(queries):
        top = np.argsort(-sims[row])[:depth]
        for rank, idx in enumerate(top, start=1):
            results.append(
                {
                    "query_id": query["query_id"],
                    "product_id": product_ids[idx],
                    "rank": rank,
                    "score": float(sims[row, idx]),
                }
            )

    del model
    torch.cuda.empty_cache()
    return results, index_time


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_pool(
    runs: dict[str, list[dict]], queries: list[dict], products: pd.DataFrame
) -> pd.DataFrame:
    """Union of every system's results, one row per (query, product)."""
    seen: dict[tuple[str, str], set[str]] = {}
    for system, rows in runs.items():
        for row in rows:
            key = (row["query_id"], row["product_id"])
            seen.setdefault(key, set()).add(system)

    query_text = {q["query_id"]: q["query"] for q in queries}
    meta = products.set_index("product_id")

    records = []
    for (query_id, product_id), systems in seen.items():
        product = meta.loc[product_id]
        records.append(
            {
                "query_id": query_id,
                "query": query_text[query_id],
                "product_id": product_id,
                "title": product["title"],
                "brand": product["brand"],
                "price": product["price"],
                "rate": product["rate"],
                "rate_count": product["rate_count"],
                "systems": ",".join(sorted(systems)),
                "relevance": "",
            }
        )
    return pd.DataFrame(records).sort_values(["query_id", "title"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval labelling pool.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    products = pd.read_parquet(args.sample)
    products["product_id"] = products["product_id"].astype(str)
    queries = load_queries(args.queries)
    product_ids = products["product_id"].tolist()
    print(f"{len(products)} products, {len(queries)} queries")

    dense_texts = products.apply(build_dense_text, axis=1).tolist()

    runs: dict[str, list[dict]] = {}
    timings: dict[str, float] = {}

    rows, elapsed = run_bm25(
        products["search_text"].tolist(), queries, product_ids, args.depth
    )
    runs["bm25"] = rows
    timings["bm25"] = elapsed
    print(f"bm25 indexed in {elapsed:.1f}s")

    if not args.skip_dense:
        for spec in MODELS:
            rows, elapsed = run_dense(
                spec,
                dense_texts,
                queries,
                product_ids,
                args.depth,
                args.out_dir / "emb_cache",
            )
            runs[spec.name] = rows
            timings[spec.name] = elapsed
            print(f"{spec.name} encoded in {elapsed:.1f}s")

    for system, rows in runs.items():
        write_jsonl(rows, args.out_dir / "runs" / f"{system}_v1.jsonl")
    write_jsonl(
        [{"system": k, "index_seconds": round(v, 2)} for k, v in timings.items()],
        args.out_dir / "runs" / "timings_v1.jsonl",
    )

    pool = build_pool(runs, queries, products)
    pool_path = args.out_dir / "pool_v1_to_label.csv"
    pool.to_csv(pool_path, index=False, encoding="utf-8-sig")

    per_query = pool.groupby("query_id").size()
    print(f"\npool rows      {len(pool)}")
    print(f"per query      min {per_query.min()}, median {int(per_query.median())}, max {per_query.max()}")
    print(f"written -> {pool_path}")


if __name__ == "__main__":
    main()