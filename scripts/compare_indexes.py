"""Find an index that is small enough to share without losing the answer.

Owner: Ali. IVF-PQ with inner product returned only a third of the exact
top-10, and raising nprobe barely moved it, which points at the quantiser
rather than the cluster search. This script tries the alternatives on equal
terms.

Vectors are read back out of the flat index with reconstruct_n instead of
being re-encoded, so trying another variant costs seconds rather than the
13 minutes the GPU pass takes.

All variants use L2. For unit-length vectors L2 and inner product induce the
same ranking, since ||a−b||² = 2 − 2·cos(a,b), and FAISS's PQ path is built
around L2, so the conversion is free and removes a known source of loss.

    python -m scripts.compare_indexes --queries data/eval/queries_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.products import DenseRetriever, load_faiss

VARIANTS: dict[str, str] = {
    "ivfpq": "IVF4096,PQ96",
    "opq_ivfpq": "OPQ96_768,IVF4096,PQ96",
    "ivfsq8": "IVF4096,SQ8",
    "ivfpq_fine": "IVF4096,PQ192x4",
}
NPROBE = 32
TRAIN_SAMPLE = 200_000


def load_vectors(index_type: str = "flat") -> np.ndarray:
    index = load_faiss(index_type)
    print(f"reconstructing {index.ntotal} vectors from the {index_type} index")
    return index.reconstruct_n(0, index.ntotal)


def build_variant(factory: str, vectors: np.ndarray, seed: int = 42):
    import faiss

    index = faiss.index_factory(vectors.shape[1], factory, faiss.METRIC_L2)
    sample = vectors
    if len(vectors) > TRAIN_SAMPLE:
        rng = np.random.default_rng(seed)
        sample = vectors[rng.choice(len(vectors), TRAIN_SAMPLE, replace=False)]
    start = time.perf_counter()
    index.train(sample)
    index.add(vectors)
    return index, time.perf_counter() - start


def index_size_mb(index) -> float:
    import faiss

    with tempfile.NamedTemporaryFile(suffix=".faiss") as handle:
        faiss.write_index(index, handle.name)
        return round(Path(handle.name).stat().st_size / 1e6, 1)


def recall_against(exact: np.ndarray, approx: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)]))


def main() -> None:
    import faiss

    parser = argparse.ArgumentParser(description="Compare FAISS index variants.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/eval/index_variants_v1.csv"))
    args = parser.parse_args()

    with args.queries.open(encoding="utf-8") as handle:
        queries = [json.loads(line)["query"] for line in handle if line.strip()]
    retriever = DenseRetriever()
    query_vectors = np.vstack([retriever.encode_query(q) for q in queries])

    vectors = load_vectors("flat")
    exact_index = faiss.IndexFlatL2(vectors.shape[1])
    exact_index.add(vectors)
    start = time.perf_counter()
    _, exact = exact_index.search(query_vectors, args.top_k)
    exact_ms = 1000 * (time.perf_counter() - start) / len(queries)
    print(f"exact: {exact_ms:.1f} ms/query, {index_size_mb(exact_index)} MB")

    rows = []
    for name, factory in VARIANTS.items():
        index, build_seconds = build_variant(factory, vectors)
        if hasattr(index, "nprobe"):
            index.nprobe = NPROBE
        start = time.perf_counter()
        _, approx = index.search(query_vectors, args.top_k)
        ms = 1000 * (time.perf_counter() - start) / len(queries)
        rows.append(
            {
                "variant": name,
                "factory": factory,
                f"recall@{args.top_k}": round(recall_against(exact, approx), 4),
                "size_mb": index_size_mb(index),
                "ms_per_query": round(ms, 3),
                "build_seconds": round(build_seconds, 1),
            }
        )
        print(rows[-1])
        del index

    report = pd.DataFrame(rows).sort_values(f"recall@{args.top_k}", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print("\n" + report.to_string(index=False))
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()