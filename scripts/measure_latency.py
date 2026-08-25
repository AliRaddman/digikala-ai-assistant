"""Measure what a retrieval call actually costs.

Owner: Ali. The index benchmarks reported FAISS search alone (0.16 ms), which
is not what a caller experiences. This measures `retrieve()` end to end, and
splits out query encoding, because on short queries the encoder — not the
search — is where the time goes.

Cold start is reported separately: the first call pays for loading the FAISS
index, the BM25 matrix and the encoder. Chains that build a retriever per
request would pay that every time, which is why the loaders are cached.

    python -m scripts.measure_latency --queries data/eval/queries_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.base import RetrievalFilters
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.products import BM25Retriever, DenseRetriever

WARMUP = 3


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    values = np.array(samples_ms)
    return {
        "mean_ms": round(float(values.mean()), 1),
        "p50_ms": round(float(np.percentile(values, 50)), 1),
        "p95_ms": round(float(np.percentile(values, 95)), 1),
        "max_ms": round(float(values.max()), 1),
    }


def time_calls(fn, queries: list[str], filters: RetrievalFilters | None) -> list[float]:
    for query in queries[:WARMUP]:
        fn(query, filters)
    samples = []
    for query in queries:
        start = time.perf_counter()
        fn(query, filters)
        samples.append(1000 * (time.perf_counter() - start))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval latency.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/eval/latency_v1.csv"))
    args = parser.parse_args()

    with args.queries.open(encoding="utf-8") as handle:
        queries = [json.loads(line)["query"] for line in handle if line.strip()]

    cold_start = time.perf_counter()
    dense = DenseRetriever()
    dense.retrieve(queries[0], top_k=args.top_k)
    dense_cold = time.perf_counter() - cold_start

    cold_start = time.perf_counter()
    sparse = BM25Retriever()
    sparse.retrieve(queries[0], top_k=args.top_k)
    sparse_cold = time.perf_counter() - cold_start

    hybrid = HybridRetriever()
    print(f"cold start   dense {dense_cold:.1f}s   bm25 {sparse_cold:.1f}s")

    tight = RetrievalFilters(price_max=1_000_000, exclude_fake=True, min_rate_count=10)

    rows = []
    scenarios = [
        ("encode only", lambda q, f: dense.encode_query(q), None),
        ("bm25", lambda q, f: sparse.retrieve(q, args.top_k, f), None),
        ("dense", lambda q, f: dense.retrieve(q, args.top_k, f), None),
        ("hybrid", lambda q, f: hybrid.retrieve(q, args.top_k, f), None),
        ("dense + filters", lambda q, f: dense.retrieve(q, args.top_k, f), tight),
        ("hybrid + filters", lambda q, f: hybrid.retrieve(q, args.top_k, f), tight),
        ("hybrid top_k=50", lambda q, f: hybrid.retrieve(q, 50, f), None),
    ]

    for name, fn, filters in scenarios:
        samples = time_calls(fn, queries, filters)
        rows.append({"scenario": name, **percentiles(samples)})
        print(rows[-1])

    report = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print("\n" + report.to_string(index=False))
    print(f"\nqueries: {len(queries)}, warmup: {WARMUP}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()