"""Measure the three retrieval failure modes instead of describing them.

Owner: Ali. Each case was found by reading real output; this puts a number on
how often it happens and, where there is a candidate fix, what the fix costs.

    python -m scripts.analyze_failures \
        --runs data/eval_d50/runs \
        --qrels data/eval/qrels_d50_v2_labeled.csv \
        --sample data/processed/products_sample_50k_v1.parquet \
        --queries data/eval/queries_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.normalize import to_search_text, tokenize
from src.eval.retrieval_metrics import evaluate_run, load_qrels, load_run
from src.retrieval.products import STOPWORDS

TOP_K = 10


def title_key(title: str) -> str:
    """Collapse titles that differ only by a model or colour code.

    Digikala titles repeat the same product with a trailing code, so exact
    string equality misses most of the duplication.
    """
    tokens = [t for t in tokenize(title) if not any(c.isdigit() for c in t)]
    drop = {"مدل", "کد", "کالای", "طرح", "سری"}
    return " ".join(t for t in tokens if t not in drop)


def case_duplicates(run, qrels, titles: dict[str, str], k: int) -> dict:
    """How much of a top-10 is the same product listed more than once."""
    distinct_ratios, duplicate_queries = [], 0
    for query_id, docs in run.items():
        keys = [title_key(titles.get(doc, "")) for doc in docs[:k]]
        keys = [key for key in keys if key]
        if not keys:
            continue
        distinct = len(set(keys))
        distinct_ratios.append(distinct / len(keys))
        if distinct < len(keys):
            duplicate_queries += 1

    deduped = {}
    for query_id, docs in run.items():
        seen, kept = set(), []
        for doc in docs:
            key = title_key(titles.get(doc, ""))
            if key and key in seen:
                continue
            seen.add(key)
            kept.append(doc)
            if len(kept) == k:
                break
        deduped[query_id] = kept

    before = evaluate_run(run, qrels, k=k, min_relevance=1)
    after = evaluate_run(deduped, qrels, k=k, min_relevance=1)
    return {
        "queries_with_duplicates": duplicate_queries,
        "queries_total": len(distinct_ratios),
        "mean_distinct_ratio": round(float(np.mean(distinct_ratios)), 3),
        f"ndcg@{k}_before": round(before[f"ndcg@{k}"].mean(), 4),
        f"ndcg@{k}_after_dedup": round(after[f"ndcg@{k}"].mean(), 4),
    }


def case_no_perfect_match(qrels) -> list[str]:
    """Queries where nothing in the judged pool is fully relevant."""
    return [
        query_id
        for query_id, judgements in qrels.items()
        if max(judgements.values(), default=0) < 2
    ]


def case_stopwords(sample: pd.DataFrame, queries: list[dict], k: int) -> pd.DataFrame:
    """What conversational tokens do to BM25.

    Words like «می‌خوام» and «باشه» almost never appear in a product title, so
    their idf is enormous and any title that happens to contain one outranks
    the products the user actually asked about.
    """
    from rank_bm25 import BM25Okapi

    index = BM25Okapi([doc.split() for doc in sample["search_text"]])
    titles = sample["title"].tolist()

    rows = []
    for query in queries:
        raw = tokenize(query["query"])
        filtered = [t for t in raw if t not in STOPWORDS] or raw
        top_raw = np.argsort(-index.get_scores(raw))[:1]
        top_filtered = np.argsort(-index.get_scores(filtered))[:1]
        if top_raw[0] != top_filtered[0]:
            rows.append(
                {
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "top1_without_stopword_removal": titles[top_raw[0]][:60],
                    "top1_with_stopword_removal": titles[top_filtered[0]][:60],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify retrieval failures.")
    parser.add_argument("--runs", type=Path, default=Path("data/eval_d50/runs"))
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    sample = pd.read_parquet(args.sample)
    sample["product_id"] = sample["product_id"].astype(str)
    titles = dict(zip(sample["product_id"], sample["title"]))
    with args.queries.open(encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle if line.strip()]

    run = load_run(args.runs / "e5-base_v1.jsonl")

    print("=== case 1: duplicate products in the same result page")
    duplicates = case_duplicates(run, qrels, titles, args.k)
    for key, value in duplicates.items():
        print(f"  {key:<28} {value}")

    print("\n=== case 2: queries with no fully relevant product in the data")
    empty = case_no_perfect_match(qrels)
    for query_id in empty:
        print(f"  {query_id}")
    print(f"  {len(empty)} of {len(qrels)} queries")

    print("\n=== case 3: conversational tokens hijacking BM25")
    stopword_df = case_stopwords(sample, queries, args.k)
    print(f"  top-1 changed on {len(stopword_df)} of {len(queries)} queries")
    if len(stopword_df):
        print(stopword_df.head(5).to_string(index=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([duplicates]).to_csv(args.out_dir / "failure_duplicates_v1.csv", index=False)
    stopword_df.to_csv(args.out_dir / "failure_stopwords_v1.csv", index=False, encoding="utf-8-sig")
    print(f"\nwritten -> {args.out_dir}")


if __name__ == "__main__":
    main()