"""Run the reproducible product-comparison evaluation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chains.product_comparison import ProductComparisonChain
from src.eval.grounding import LLMGroundingJudge
from src.eval.product_comparison import (
    ProductComparisonEvaluator,
    load_comparison_cases,
)
from src.llm.client import build_openai_client
from src.llm.config import LLMSettings
from src.llm.usage import SQLiteUsageLedger
from src.retrieval.base import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ProductComparisonChain. Retrieval-only is the default; "
            "live LLM calls require an explicit --with-llm flag."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/eval/product_comparison_cases_v1.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retriever-mode", choices=("mock", "real"), default="real")
    parser.add_argument("--comment-top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Generate comparison inference. This may make paid API calls.",
    )
    parser.add_argument(
        "--judge-grounding",
        action="store_true",
        help="Judge generated comparisons. Requires --with-llm and may cost money.",
    )
    args = parser.parse_args()

    if args.judge_grounding and not args.with_llm:
        parser.error("--judge-grounding requires --with-llm")
    if args.comment_top_k < 1:
        parser.error("--comment-top-k must be at least 1")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be at least 1")

    settings = LLMSettings.from_env()
    llm_client = build_openai_client(settings) if args.with_llm else None
    ledger = SQLiteUsageLedger(settings.usage_path) if llm_client else None
    judge = (
        LLMGroundingJudge(llm_client)
        if args.judge_grounding and llm_client
        else None
    )
    cases = load_comparison_cases(args.input)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    evaluator = ProductComparisonEvaluator(
        chain=ProductComparisonChain(
            product_retriever=build_retriever("product", mode=args.retriever_mode),
            comment_retriever=build_retriever("comment", mode=args.retriever_mode),
            llm_client=llm_client,
            comment_top_k=args.comment_top_k,
        ),
        ledger=ledger,
        grounding_judge=judge,
    )
    report = evaluator.evaluate(cases)
    report["configuration"].update(
        {
            "input": str(args.input),
            "retriever_mode": args.retriever_mode,
            "max_cases": args.max_cases,
        }
    )

    output = args.output or Path(
        "data/eval/runs/"
        + (
            "product_comparison_llm_judged_v1.json"
            if args.judge_grounding
            else "product_comparison_llm_v1.json"
            if args.with_llm
            else "product_comparison_retrieval_v1.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
