"""Is the hybrid improvement real, or is it 36 queries of noise?

Owner: Ali. Fusion raised nDCG@10 from 0.6916 to 0.6994, which is +1.1%. On
36 queries that is well inside the range a couple of lucky queries could
produce, so the difference gets tested rather than announced.

Both tests are paired: the same query is scored under both systems and only
the per-query difference is examined, which removes the fact that some
queries are simply harder than others.

    bootstrap: resample the 36 query-level differences with replacement
               10,000 times; report how often the mean difference flips sign
    t-test:    parametric equivalent, included as a cross-check

    python -m scripts.test_significance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.retrieval_metrics import evaluate_run, load_qrels, load_run
from scripts.eval_hybrid import fuse_rrf

BOOTSTRAP_SAMPLES = 10_000
SEED = 42


def per_query_scores(run: dict[str, list[str]], qrels, k: int) -> pd.DataFrame:
    return evaluate_run(run, qrels, k=k, min_relevance=1).set_index("query_id")


def paired_bootstrap(
    diffs: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED
) -> tuple[float, tuple[float, float]]:
    """Two-sided p-value and 95% confidence interval of the mean difference.

    The p-value is the share of resamples whose mean lands on the other side
    of zero, doubled for a two-sided test.
    """
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(samples)]
    )
    observed = diffs.mean()
    tail = float((means <= 0).mean() if observed > 0 else (means >= 0).mean())
    return min(1.0, 2 * tail), (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def compare(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame, metric: str) -> dict:
    common = a.index.intersection(b.index)
    diffs = (b.loc[common, metric] - a.loc[common, metric]).to_numpy()
    p_boot, (low, high) = paired_bootstrap(diffs)
    _, p_t = stats.ttest_rel(b.loc[common, metric], a.loc[common, metric])
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    return {
        "metric": metric,
        "baseline": name_a,
        "system": name_b,
        "mean_baseline": round(float(a.loc[common, metric].mean()), 4),
        "mean_system": round(float(b.loc[common, metric].mean()), 4),
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "p_bootstrap": round(p_boot, 4),
        "p_ttest": round(float(p_t), 4),
        "wins": wins,
        "losses": losses,
        "ties": len(diffs) - wins - losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether hybrid beats dense.")
    parser.add_argument("--runs", type=Path, default=Path("data/eval/runs"))
    parser.add_argument("--qrels", type=Path, default=Path("data/eval/qrels_v1_labeled.csv"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--w-dense", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("data/eval/significance_v1.csv"))
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    dense = load_run(args.runs / "e5-base_v1.jsonl")
    sparse = load_run(args.runs / "bm25_v1.jsonl")
    hybrid = fuse_rrf(
        {"dense": dense, "sparse": sparse},
        {"dense": args.w_dense, "sparse": 1 - args.w_dense},
        k=args.rrf_k,
        top_k=args.k,
    )

    scores = {
        "dense": per_query_scores(dense, qrels, args.k),
        "bm25": per_query_scores(sparse, qrels, args.k),
        "hybrid": per_query_scores(hybrid, qrels, args.k),
    }

    rows = []
    for metric in (f"ndcg@{args.k}", f"recall@{args.k}", f"mrr@{args.k}"):
        rows.append(compare("dense", scores["dense"], "hybrid", scores["hybrid"], metric))
        rows.append(compare("bm25", scores["bm25"], "hybrid", scores["hybrid"], metric))

    report = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))

    print(f"\nqueries: {len(scores['dense'])}")
    print("p < 0.05 means the difference is unlikely to be resampling noise")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()