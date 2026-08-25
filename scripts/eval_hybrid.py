"""Does combining lexical and semantic retrieval actually help?

Owner: Ali. Fusion happens at the rank level, so this reuses the run files
already produced by build_pool instead of re-encoding anything — the question
is only how to merge two orderings that already exist.

Two fusion methods are compared:

    RRF:      score(d) = Σ_i w_i / (k + rank_i(d))
    weighted: score(d) = α · norm(dense_d) + (1 − α) · norm(bm25_d)

RRF ignores the raw scores and uses only positions, which sidesteps the fact
that BM25 scores are unbounded (they ran to 20+ here) while cosine similarity
sits in a narrow band near 0.85. The weighted variant is included to show
whether that scale problem is real or theoretical.

    python -m scripts.eval_hybrid \
        --runs data/eval/runs --qrels data/eval/qrels_v1_labeled.csv

Caveat: the runs hold each system's top-10 on the 50k benchmark sample, and
qrels only judged that pool, so fusion cannot surface anything neither system
retrieved. The comparison between fused and single systems stays fair, but
the absolute numbers are pooled, not global.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.retrieval_metrics import evaluate_run, load_qrels, load_run

DEFAULT_K = 10
RRF_K_VALUES = (10, 20, 60)
DENSE_WEIGHTS = (0.3, 0.5, 0.7)
ALPHA_VALUES = tuple(round(0.1 * i, 1) for i in range(11))


def load_scored_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """query_id -> [(product_id, score)] in rank order."""
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    run: dict[str, list[tuple[str, float]]] = {}
    for row in sorted(rows, key=lambda r: (r["query_id"], r["rank"])):
        run.setdefault(row["query_id"], []).append(
            (str(row["product_id"]), float(row["score"]))
        )
    return run


def fuse_rrf(
    runs: dict[str, dict[str, list[str]]],
    weights: dict[str, float],
    k: int,
    top_k: int,
) -> dict[str, list[str]]:
    fused: dict[str, list[str]] = {}
    query_ids = {qid for run in runs.values() for qid in run}
    for query_id in query_ids:
        scores: dict[str, float] = {}
        for system, run in runs.items():
            for rank, product_id in enumerate(run.get(query_id, []), start=1):
                scores[product_id] = scores.get(product_id, 0.0) + weights[system] / (
                    k + rank
                )
        fused[query_id] = [
            pid for pid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        ]
    return fused


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def fuse_weighted(
    dense: dict[str, list[tuple[str, float]]],
    sparse: dict[str, list[tuple[str, float]]],
    alpha: float,
    top_k: int,
) -> dict[str, list[str]]:
    fused: dict[str, list[str]] = {}
    for query_id in set(dense) | set(sparse):
        scores: dict[str, float] = {}
        for run, weight in ((dense, alpha), (sparse, 1 - alpha)):
            items = run.get(query_id, [])
            normalised = _min_max([score for _, score in items])
            for (product_id, _), value in zip(items, normalised):
                scores[product_id] = scores.get(product_id, 0.0) + weight * value
        fused[query_id] = [
            pid for pid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        ]
    return fused


def score(run: dict[str, list[str]], qrels, k: int) -> dict[str, float]:
    lenient = evaluate_run(run, qrels, k=k, min_relevance=1)
    strict = evaluate_run(run, qrels, k=k, min_relevance=2)
    return {
        f"ndcg@{k}": round(lenient[f"ndcg@{k}"].mean(), 4),
        f"recall@{k}": round(lenient[f"recall@{k}"].mean(), 4),
        f"recall@{k}_strict": round(strict[f"recall@{k}"].mean(), 4),
        f"mrr@{k}": round(lenient[f"mrr@{k}"].mean(), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid fusion.")
    parser.add_argument("--runs", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--qrels", type=Path, default=Path("data/eval/qrels_v1_labeled.csv"))
    parser.add_argument("--dense", default="e5-base")
    parser.add_argument("--sparse", default="bm25")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=Path("data/eval/hybrid_v1.csv"))
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    dense_scored = load_scored_run(args.runs / f"{args.dense}_v1.jsonl")
    sparse_scored = load_scored_run(args.runs / f"{args.sparse}_v1.jsonl")
    dense = load_run(args.runs / f"{args.dense}_v1.jsonl")
    sparse = load_run(args.runs / f"{args.sparse}_v1.jsonl")

    rows = [
        {"method": "dense only", "param": args.dense, **score(dense, qrels, args.k)},
        {"method": "sparse only", "param": args.sparse, **score(sparse, qrels, args.k)},
    ]

    for rrf_k in RRF_K_VALUES:
        for weight in DENSE_WEIGHTS:
            fused = fuse_rrf(
                {"dense": dense, "sparse": sparse},
                {"dense": weight, "sparse": 1 - weight},
                k=rrf_k,
                top_k=args.k,
            )
            rows.append(
                {
                    "method": "rrf",
                    "param": f"k={rrf_k}, w_dense={weight}",
                    **score(fused, qrels, args.k),
                }
            )

    for alpha in ALPHA_VALUES:
        fused = fuse_weighted(dense_scored, sparse_scored, alpha, top_k=args.k)
        rows.append(
            {
                "method": "weighted",
                "param": f"alpha={alpha}",
                **score(fused, qrels, args.k),
            }
        )

    report = pd.DataFrame(rows)
    report = report.sort_values(f"ndcg@{args.k}", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)

    print(report.to_string(index=False))

    baseline = float(report.loc[report["method"] == "dense only", f"ndcg@{args.k}"].iloc[0])
    best = report.iloc[0]
    delta = 100 * (best[f"ndcg@{args.k}"] - baseline) / baseline
    print(f"\nbest: {best['method']} ({best['param']})")
    print(f"nDCG@{args.k} {baseline} -> {best[f'ndcg@{args.k}']}  ({delta:+.1f}%)")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()