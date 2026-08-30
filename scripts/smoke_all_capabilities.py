"""End-to-end smoke test of all four mandatory capabilities, through the orchestrator.

Owner: Ali, 2026-08-30. One command that routes four real Persian requests and
prints the status of each, so the state of the whole system is visible at a
glance before a demo rather than one chain at a time.

Every capability is exercised through ShoppingAssistantOrchestrator.run, not by
calling the chains directly: a chain that works but is not registered still
fails here, which is exactly the failure this script exists to catch.

    python scripts/smoke_all_capabilities.py                    # mock, no key
    python scripts/smoke_all_capabilities.py --retriever-mode real
    python scripts/smoke_all_capabilities.py --retriever-mode real --answers

--no-llm forces the offline path. Section 2 needs a model and reports
`llm_unavailable` without one; sections 1, 3 and 4 answer regardless.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from src.orchestrator import build_default_orchestrator

# Two ear-swab products from the same cat1 (بهداشت و مراقبت بدن), 50 indexed
# reviews each, chosen so the comparison has something to say:
#
#   340807  گوش پاک کن چرخشی اسمارت — 55,000 تومان — 3 recommended / 33 not,
#           mean rate 2.06. The dissatisfied one, and the expensive one.
#   5693178 گوش پاک کن نازنیکا      — 29,800 تومان — 33 recommended / 0 not,
#           mean rate 4.15. The satisfied one, and the cheap one.
#
# Deliberately the pricier product that reviewers like less, so the answer
# cannot be read off the price. Rates count only comments carrying an explicit
# recommendation_status: treating NaN as "not recommended" inverted the ranking
# during selection -- the same mistake as failure 5 in docs/FAILURES.md.
#
# The dissatisfied product is first because section 2 asks REAL_IDS[0] what its
# recurring complaints are.
# In mock mode these are unknown, and the mock ids below are used instead.
REAL_IDS = ("340807", "5693178")
MOCK_IDS = ("3901234", "6604311")


def _cases(product_ids: tuple[str, str]) -> list[dict[str, Any]]:
    first, second = product_ids
    return [
        {
            "section": "۱ — جست‌وجو و پیشنهاد محصول",
            "expected_intent": "product_discovery",
            "query": "یه کوله پشتی مناسب مدرسه می‌خوام که جادار باشه",
            "kwargs": {},
        },
        {
            "section": "۲ — پرسش و پاسخ بر پایه نظرات",
            "expected_intent": "product_qa",
            "query": "ایرادهای پرتکرار این محصول چیست؟",
            "kwargs": {"product_ids": [first]},
        },
        {
            "section": "۳ — مقایسه دو محصول",
            "expected_intent": "product_comparison",
            "query": "این دو محصول را مقایسه کن",
            "kwargs": {"product_ids": [first, second]},
        },
        {
            "section": "۴ — تحلیل دسته برای مدیر فروشگاه",
            "expected_intent": "category_analytics",
            "query": "پرتکرارترین شکایت کاربران در دسته اسباب بازی چیست؟",
            "kwargs": {},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-mode", default="mock")
    parser.add_argument("--answers", action="store_true",
                        help="print the Persian answer of each capability, not just its status")
    parser.add_argument("--chars", type=int, default=700)
    parser.add_argument("--no-llm", action="store_true",
                        help="blank LLM_API_KEY so nothing can reach the network")
    args = parser.parse_args()

    if args.no_llm:
        os.environ["LLM_API_KEY"] = ""

    orchestrator = build_default_orchestrator(retriever_mode=args.retriever_mode)
    ids = MOCK_IDS if args.retriever_mode == "mock" else REAL_IDS

    rows = []
    for case in _cases(ids):
        started = time.perf_counter()
        result = orchestrator.run(case["query"], **case["kwargs"])
        elapsed = (time.perf_counter() - started) * 1000

        routed_right = result.route.intent == case["expected_intent"]
        rows.append(
            {
                "section": case["section"],
                "query": case["query"],
                "status": result.status,
                "intent": result.route.intent,
                "routed_right": routed_right,
                "citations": len(result.citations),
                "chars": len(result.answer),
                "ms": elapsed,
                "error": result.error,
                "answer": result.answer,
            }
        )

    print(f"\nretriever_mode={args.retriever_mode}  product_ids={ids}\n")
    header = f"{'بخش':38} {'status':22} {'intent':20} {'route':6} {'cites':>5} {'ms':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['section']:38} {row['status']:22} {row['intent']:20} "
            f"{'ok' if row['routed_right'] else 'WRONG':6} "
            f"{row['citations']:>5} {row['ms']:>9.1f}"
        )
        if row["error"]:
            print(f"    error: {row['error']}")

    if args.answers:
        for row in rows:
            print("\n" + "=" * 78)
            print(f"{row['section']}   —   {row['query']}")
            print("=" * 78)
            print(row["answer"][: args.chars])
            if row["chars"] > args.chars:
                print(f"... [{row['chars'] - args.chars} کاراکتر دیگر]")

    ok = sum(r["status"] == "success" and r["routed_right"] for r in rows)
    print(f"\n{ok}/{len(rows)} capabilities answered through the orchestrator.")
    raise SystemExit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
