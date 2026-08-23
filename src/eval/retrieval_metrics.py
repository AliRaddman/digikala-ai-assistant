"""Retrieval metrics for the embedding benchmark.

Owner: Ali.

    python -m src.eval.retrieval_metrics \
        --qrels data/eval/qrels_v1_labeled.csv \
        --runs data/eval/runs \
        --queries data/eval/queries_v1.jsonl

Relevance is graded 0/1/2. Two views are reported: `lenient` counts anything
above 0 as relevant, `strict` only counts 2. A model can look strong under
lenient scoring by returning near-misses, so the gap between the two columns
is where the models actually separate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_K = 10


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    df = pd.read_csv(path)
    df["product_id"] = df["product_id"].astype(str)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in df.itertuples():
        qrels[row.query_id][row.product_id] = int(row.relevance)
    return dict(qrels)


def load_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in sorted(rows, key=lambda r: (r["query_id"], r["rank"])):
        run[row["query_id"]].append(str(row["product_id"]))
    return dict(run)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of known relevant items that appear in the top k.

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    The denominator comes from the judged pool, not the whole catalogue, so
    this is pooled recall: it answers "of the items any system found and a
    human judged relevant, how many did this system rank in its top k".
    """
    if not relevant:
        return float("nan")
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def dcg(gains: list[float]) -> float:
    """DCG with the standard exponential gain and log2 discount.

    DCG@k = Σ_{i=1..k} (2^rel_i − 1) / log2(i + 1)
    """
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: list[str], judgements: dict[str, int], k: int) -> float:
    """DCG of the run divided by DCG of the perfect ranking.

    nDCG@k = DCG@k / IDCG@k

    Graded labels matter here: a system that puts every 2 above every 1 scores
    higher than one that merely retrieves the same set in any order, which is
    exactly the difference Recall cannot see.
    """
    gains = [float(judgements.get(pid, 0)) for pid in retrieved[:k]]
    ideal = sorted((float(v) for v in judgements.values()), reverse=True)[:k]
    denominator = dcg(ideal)
    if denominator == 0:
        return float("nan")
    return dcg(gains) / denominator


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit.

    MRR@k = 1 / rank_first_relevant   (0 when there is no hit in the top k)
    """
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = DEFAULT_K,
    min_relevance: int = 1,
) -> pd.DataFrame:
    """Per-query metrics for one system."""
    records = []
    for query_id, judgements in qrels.items():
        retrieved = run.get(query_id, [])
        relevant = {pid for pid, rel in judgements.items() if rel >= min_relevance}
        records.append(
            {
                "query_id": query_id,
                f"recall@{k}": recall_at_k(retrieved, relevant, k),
                f"ndcg@{k}": ndcg_at_k(retrieved, judgements, k),
                f"mrr@{k}": mrr_at_k(retrieved, relevant, k),
            }
        )
    return pd.DataFrame(records)


def summarise(
    runs_dir: Path,
    qrels: dict[str, dict[str, int]],
    queries: list[dict],
    k: int = DEFAULT_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    intent_of = {q["query_id"]: q.get("intent", "unknown") for q in queries}

    overall_rows = []
    per_intent_rows = []

    for path in sorted(runs_dir.glob("*_v1.jsonl")):
        system = path.stem.replace("_v1", "")
        if system == "timings":
            continue
        run = load_run(path)

        lenient = evaluate_run(run, qrels, k=k, min_relevance=1)
        strict = evaluate_run(run, qrels, k=k, min_relevance=2)

        overall_rows.append(
            {
                "system": system,
                f"recall@{k}": round(lenient[f"recall@{k}"].mean(), 4),
                f"recall@{k}_strict": round(strict[f"recall@{k}"].mean(), 4),
                f"ndcg@{k}": round(lenient[f"ndcg@{k}"].mean(), 4),
                f"mrr@{k}": round(lenient[f"mrr@{k}"].mean(), 4),
            }
        )

        lenient["intent"] = lenient["query_id"].map(intent_of)
        grouped = lenient.groupby("intent")[[f"recall@{k}", f"ndcg@{k}"]].mean().round(4)
        for intent, row in grouped.iterrows():
            per_intent_rows.append(
                {
                    "system": system,
                    "intent": intent,
                    f"recall@{k}": row[f"recall@{k}"],
                    f"ndcg@{k}": row[f"ndcg@{k}"],
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values(f"ndcg@{k}", ascending=False)
    per_intent = pd.DataFrame(per_intent_rows).pivot(
        index="intent", columns="system", values=f"ndcg@{k}"
    )
    return overall, per_intent


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the retrieval runs.")
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--out", type=Path, default=Path("data/eval/metrics_v1.csv"))
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    with args.queries.open(encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle if line.strip()]

    overall, per_intent = summarise(args.runs, qrels, queries, k=args.k)

    print(overall.to_string(index=False))
    print(f"\nnDCG@{args.k} by intent")
    print(per_intent.to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out, index=False)
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()