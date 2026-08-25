"""Build the smallest labelling set that can settle the hybrid question.

Owner: Ali. A depth-50 pool across four systems is 4,464 rows, but the
hybrid experiment only compares two systems and only ever shows ten results.
What has to be judged is therefore the union of the top-10 of every candidate
fusion plus the two baselines — not the whole retrieval depth. Everything
already judged in qrels_v1 is carried over unchanged, so the two experiments
stay on one consistent set of labels.

    python -m scripts.build_hybrid_pool \
        --runs data/eval_d50/runs \
        --qrels data/eval/qrels_v1_labeled.csv \
        --sample data/processed/products_sample_50k_v1.parquet \
        --out-dir data/eval_d50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.eval_hybrid import (
    ALPHA_VALUES,
    DENSE_WEIGHTS,
    RRF_K_VALUES,
    fuse_rrf,
    fuse_weighted,
    load_scored_run,
)
from src.eval.retrieval_metrics import load_run

TOP_K = 10


def candidate_pairs(runs_dir: Path, dense_name: str, sparse_name: str) -> set[tuple[str, str]]:
    dense = load_run(runs_dir / f"{dense_name}_v1.jsonl")
    sparse = load_run(runs_dir / f"{sparse_name}_v1.jsonl")
    dense_scored = load_scored_run(runs_dir / f"{dense_name}_v1.jsonl")
    sparse_scored = load_scored_run(runs_dir / f"{sparse_name}_v1.jsonl")

    fused_runs: list[dict[str, list[str]]] = [
        {qid: docs[:TOP_K] for qid, docs in dense.items()},
        {qid: docs[:TOP_K] for qid, docs in sparse.items()},
    ]
    for rrf_k in RRF_K_VALUES:
        for weight in DENSE_WEIGHTS:
            fused_runs.append(
                fuse_rrf(
                    {"dense": dense, "sparse": sparse},
                    {"dense": weight, "sparse": 1 - weight},
                    k=rrf_k,
                    top_k=TOP_K,
                )
            )
    for alpha in ALPHA_VALUES:
        fused_runs.append(fuse_weighted(dense_scored, sparse_scored, alpha, top_k=TOP_K))

    pairs: set[tuple[str, str]] = set()
    for run in fused_runs:
        for query_id, docs in run.items():
            pairs.update((query_id, doc) for doc in docs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal labelling set for hybrid.")
    parser.add_argument("--runs", type=Path, default=Path("data/eval_d50/runs"))
    parser.add_argument("--qrels", type=Path, default=Path("data/eval/qrels_v1_labeled.csv"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval_d50"))
    parser.add_argument("--dense", default="e5-base")
    parser.add_argument("--sparse", default="bm25")
    args = parser.parse_args()

    pairs = candidate_pairs(args.runs, args.dense, args.sparse)

    known = pd.read_csv(args.qrels)
    known["product_id"] = known["product_id"].astype(str)
    known_pairs = {
        (row.query_id, row.product_id): int(row.relevance) for row in known.itertuples()
    }

    products = pd.read_parquet(args.sample)
    products["product_id"] = products["product_id"].astype(str)
    meta = products.set_index("product_id")
    query_text = dict(zip(known["query_id"], known["query"]))

    carried, new = [], []
    for query_id, product_id in sorted(pairs):
        if product_id not in meta.index:
            continue
        product = meta.loc[product_id]
        record = {
            "query_id": query_id,
            "query": query_text.get(query_id, ""),
            "product_id": product_id,
            "title": product["title"],
            "brand": product["brand"],
            "price": product["price"],
            "rate": product["rate"],
            "rate_count": product["rate_count"],
        }
        if (query_id, product_id) in known_pairs:
            record["relevance"] = known_pairs[(query_id, product_id)]
            carried.append(record)
        else:
            record["relevance"] = ""
            new.append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    carried_df = pd.DataFrame(carried)
    new_df = pd.DataFrame(new).sort_values(["query_id", "title"])

    carried_path = args.out_dir / "qrels_carried_v2.csv"
    new_path = args.out_dir / "pool_v2_to_label.csv"
    carried_df.to_csv(carried_path, index=False, encoding="utf-8-sig")
    new_df.to_csv(new_path, index=False, encoding="utf-8-sig")

    print(f"candidate pairs      {len(pairs)}")
    print(f"already labelled     {len(carried_df)}")
    print(f"need labelling       {len(new_df)}")
    if len(new_df):
        per_query = new_df.groupby("query_id").size()
        print(f"per query            min {per_query.min()}, median {int(per_query.median())}, max {per_query.max()}")
    print(f"written -> {new_path}")


if __name__ == "__main__":
    main()