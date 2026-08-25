"""Measure what quantisation costs.

Owner: Ali. The flat index is exact, so its top-k is the ground truth by
definition and no human labels are involved: the question is not "are these
results good" but "how much of the exact answer does the compressed index
still return". Reported against nprobe, because nprobe is the one knob that
trades latency for recall after the index is built.

    python -m scripts.eval_index_quality --queries data/eval/queries_v1.jsonl

Note this cannot be scored against qrels_v1: those labels were collected on
the 50k benchmark sample, while these indexes cover all 948k products, so
almost every result here is unjudged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.products import DenseRetriever, load_faiss

NPROBE_VALUES = (4, 8, 16, 32, 64, 128)


def search(index, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    _, positions = index.search(vectors, top_k)
    elapsed_ms = 1000 * (time.perf_counter() - start) / len(vectors)
    return positions, elapsed_ms


def recall_against(exact: np.ndarray, approx: np.ndarray) -> float:
    """Mean overlap of the two top-k lists.

    recall@k = (1/Q) · Σ_q |approx_q ∩ exact_q| / k
    """
    hits = [len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)]
    return float(np.mean(hits))


def main() -> None:
    parser = argparse.ArgumentParser(description="IVF-PQ vs flat index quality.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/eval/index_quality_v1.csv"))
    args = parser.parse_args()

    with args.queries.open(encoding="utf-8") as handle:
        queries = [json.loads(line)["query"] for line in handle if line.strip()]

    retriever = DenseRetriever()
    vectors = np.vstack([retriever.encode_query(q) for q in queries])
    print(f"{len(queries)} queries encoded")

    flat = load_faiss("flat")
    exact, flat_ms = search(flat, vectors, args.top_k)
    print(f"flat: {flat_ms:.1f} ms/query")

    ivfpq = load_faiss("ivfpq")
    rows = []
    for nprobe in NPROBE_VALUES:
        ivfpq.nprobe = nprobe
        approx, ms = search(ivfpq, vectors, args.top_k)
        rows.append(
            {
                "nprobe": nprobe,
                f"recall@{args.top_k}_vs_flat": round(recall_against(exact, approx), 4),
                "ms_per_query": round(ms, 2),
                "speedup_vs_flat": round(flat_ms / ms, 1),
            }
        )
        print(rows[-1])

    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)
    print(f"\nflat baseline: {flat_ms:.1f} ms/query, 2913 MB")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()