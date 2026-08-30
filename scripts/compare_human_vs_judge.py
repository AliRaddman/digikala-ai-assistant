"""Stage 5b: agreement between Ali's hand labels and the LLM judge.

Owner: Ali, 2026-08-30. Reads the filled-in data/eval/human/labels_v1.csv and
the sidecar data/eval/human/judge_scores_v1.json that was deliberately kept
out of it, and reports, per dimension:

  * Cohen's kappa, unweighted and quadratic-weighted
  * Spearman and Pearson correlation
  * every disagreement with the full text, so the source can be read
  * whether the judge is systematically harsher or softer, rather than just
    noisy -- a signed mean difference plus a Wilcoxon signed-rank test

Unweighted kappa treats 4-vs-5 as exactly as wrong as 1-vs-5, which is the
wrong model for an ordinal 1-5 rubric, so the quadratic-weighted figure is the
one to quote; the unweighted one is reported beside it because "Cohen's kappa"
without qualification usually means that, and hiding it would be convenient.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DIMENSIONS = (("grounding_1_5", "grounding"), ("relevance_1_5", "relevance"))


def _load(labels: Path, judge: Path):
    rows = list(csv.DictReader(labels.open(encoding="utf-8-sig")))
    scores = json.loads(judge.read_text(encoding="utf-8"))["scores"]
    paired, unlabelled = [], []
    for row in rows:
        entry = scores.get(row["id"])
        if entry is None:
            continue
        human = {}
        for column, name in DIMENSIONS:
            value = (row.get(column) or "").strip()
            human[name] = int(value) if value else None
        if any(v is None for v in human.values()):
            unlabelled.append(row["id"])
            continue
        paired.append({"row": row, "human": human, "judge": entry})
    return paired, unlabelled


def _agreement(human: list[int], judge: list[int]) -> dict[str, object]:
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import pearsonr, spearmanr, wilcoxon

    labels = sorted(set(human) | set(judge))
    result: dict[str, object] = {
        "n": len(human),
        "exact_agreement": sum(h == j for h, j in zip(human, judge)) / len(human),
        "within_one": sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / len(human),
        "human_mean": sum(human) / len(human),
        "judge_mean": sum(judge) / len(judge),
    }
    result["mean_difference_human_minus_judge"] = (
        result["human_mean"] - result["judge_mean"]
    )
    if len(labels) < 2:
        result["kappa"] = None
        result["kappa_quadratic"] = None
        result["kappa_note"] = (
            "undefined: every score on both sides is identical, so there is no "
            "variation for chance agreement to be corrected against"
        )
    else:
        result["kappa"] = cohen_kappa_score(human, judge, labels=labels)
        result["kappa_quadratic"] = cohen_kappa_score(
            human, judge, labels=labels, weights="quadratic"
        )
    try:
        result["spearman"] = float(spearmanr(human, judge).statistic)
    except Exception:
        result["spearman"] = None
    try:
        result["pearson"] = float(pearsonr(human, judge).statistic)
    except Exception:
        result["pearson"] = None

    harsher = sum(j < h for h, j in zip(human, judge))
    softer = sum(j > h for h, j in zip(human, judge))
    result["judge_harsher_count"] = harsher
    result["judge_softer_count"] = softer
    differences = [h - j for h, j in zip(human, judge) if h != j]
    if differences:
        try:
            result["wilcoxon_p"] = float(wilcoxon(differences).pvalue)
        except Exception as exc:
            result["wilcoxon_p"] = None
            result["wilcoxon_note"] = str(exc)
    else:
        result["wilcoxon_p"] = None
        result["wilcoxon_note"] = "no disagreements"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/eval/human/labels_v1.csv"))
    parser.add_argument("--judge", type=Path, default=Path("data/eval/human/judge_scores_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/human/agreement_v1.json"))
    parser.add_argument("--show-text", action="store_true",
                        help="print the full question/answer/evidence of each disagreement")
    args = parser.parse_args()

    paired, unlabelled = _load(args.labels, args.judge)
    if not paired:
        raise SystemExit(
            f"no fully labelled rows in {args.labels}; fill in both score columns first"
        )

    report: dict[str, object] = {
        "labelled": len(paired),
        "unlabelled_ids": unlabelled,
        "dimensions": {},
        "disagreements": [],
    }
    for _, name in DIMENSIONS:
        report["dimensions"][name] = _agreement(
            [p["human"][name] for p in paired],
            [p["judge"][name] for p in paired],
        )

    for pair in paired:
        deltas = {
            name: pair["human"][name] - pair["judge"][name] for _, name in DIMENSIONS
        }
        if not any(deltas.values()):
            continue
        report["disagreements"].append(
            {
                "id": pair["row"]["id"],
                "source": pair["judge"]["source"],
                "human": pair["human"],
                "judge": {k: pair["judge"][k] for k in ("grounding", "relevance")},
                "delta_human_minus_judge": deltas,
                "judge_rationale": pair["judge"]["rationale"],
                "question": pair["row"]["question"],
                "answer": pair["row"]["answer"],
                "evidence": pair["row"]["evidence"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for name, stats in report["dimensions"].items():
        print(f"--- {name}  (n={stats['n']})")
        for key in ("kappa", "kappa_quadratic", "spearman", "pearson",
                    "exact_agreement", "within_one", "human_mean", "judge_mean",
                    "mean_difference_human_minus_judge",
                    "judge_harsher_count", "judge_softer_count", "wilcoxon_p"):
            value = stats.get(key)
            print(f"  {key:34} {value if value is None else round(value, 4) if isinstance(value, float) else value}")
        if stats.get("kappa_note"):
            print(f"  note: {stats['kappa_note']}")
        print()

    print(f"disagreements: {len(report['disagreements'])} of {len(paired)}")
    for item in report["disagreements"]:
        print(f"  {item['id']:6} {item['source']:12} "
              f"human g/r {item['human']['grounding']}/{item['human']['relevance']}  "
              f"judge g/r {item['judge']['grounding']}/{item['judge']['relevance']}")
        if args.show_text:
            print(f"    question: {item['question']}")
            print(f"    judge said: {item['judge_rationale']}")
            print(f"    answer:\n{item['answer']}\n")
    print(f"\nfull report: {args.output}")


if __name__ == "__main__":
    main()
