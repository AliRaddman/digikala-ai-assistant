import csv
import json
import time
from pathlib import Path

from src.chains.product_comparison import ProductComparisonChain
from src.retrieval.comments import CommentRetriever
from scripts.test_real_product_comparison import MetadataProductRetriever


PRODUCT_META = Path("data/indexes/products_meta_v1.parquet")
COMMENT_INDEX = Path("data/indexes/comparison_eval")

OUTPUT_DIR = Path("outputs/product_comparison")
SUMMARY_PATH = OUTPUT_DIR / "comparison_eval_summary.csv"
FULL_PATH = OUTPUT_DIR / "comparison_eval_results.jsonl"


# Real same-category / closely related product pairs
PAIRS = [
    ("1460474", "2136542"),  # Jaipur board games
    ("650617", "2198154"),   # gift sets
    ("81934", "130874"),     # men's fragrances
    ("81934", "61695"),      # men's fragrances
    ("130874", "61695"),     # men's fragrances
    ("144085", "3253386"),   # flashlights
]


QUERIES = [
    "این دو محصول را به طور کلی مقایسه کن و بگو کدام انتخاب بهتری است.",
    "این دو محصول را از نظر کیفیت و نقاط قوت و ضعف مقایسه کن.",
    "این دو محصول را از نظر ارزش خرید و قیمت مقایسه کن.",
    "بر اساس نظر کاربران، تجربه استفاده از این دو محصول را مقایسه کن.",
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    product_retriever = MetadataProductRetriever(PRODUCT_META)
    comment_retriever = CommentRetriever(index_dir=COMMENT_INDEX)

    chain = ProductComparisonChain(
        product_retriever=product_retriever,
        comment_retriever=comment_retriever,
        llm_client=None,
        comment_top_k=5,
    )

    rows = []
    full_results = []

    case_id = 0

    for product_1, product_2 in PAIRS:
        for query in QUERIES:

            case_id += 1

            print(
                f"[{case_id:02d}/24] "
                f"{product_1} vs {product_2}"
            )

            start = time.perf_counter()

            try:
                result = chain.run(
                    query,
                    product_ids=[
                        product_1,
                        product_2,
                    ],
                )

                latency = time.perf_counter() - start

                evidence_counts = {
                    group.product_id: len(group.comments)
                    for group in result.evidence
                }

                product_consistent = all(
                    all(
                        str(comment.product_id)
                        == str(group.product_id)
                        for comment in group.comments
                    )
                    for group in result.evidence
                )

                facts_ok = len(result.facts) == 2
                no_missing = len(result.missing_products) == 0

                evidence_available = (
                    result.as_dict()["evidence"]["available"]
                )

                passed = (
                    facts_ok
                    and no_missing
                    and evidence_available
                    and product_consistent
                )

                row = {
                    "case_id": case_id,
                    "product_1": product_1,
                    "product_2": product_2,
                    "query": query,
                    "facts_count": len(result.facts),
                    "product_1_evidence": evidence_counts.get(
                        product_1, 0
                    ),
                    "product_2_evidence": evidence_counts.get(
                        product_2, 0
                    ),
                    "missing_products": len(
                        result.missing_products
                    ),
                    "product_consistent": product_consistent,
                    "evidence_available": evidence_available,
                    "inference_available": (
                        result.inference is not None
                    ),
                    "latency_seconds": round(latency, 3),
                    "passed": passed,
                    "error": "",
                }

                full_results.append(
                    {
                        "case_id": case_id,
                        "product_ids": [
                            product_1,
                            product_2,
                        ],
                        "query": query,
                        "latency_seconds": latency,
                        "passed": passed,
                        "result": result.as_dict(),
                    }
                )

            except Exception as exc:

                latency = time.perf_counter() - start

                row = {
                    "case_id": case_id,
                    "product_1": product_1,
                    "product_2": product_2,
                    "query": query,
                    "facts_count": 0,
                    "product_1_evidence": 0,
                    "product_2_evidence": 0,
                    "missing_products": 2,
                    "product_consistent": False,
                    "evidence_available": False,
                    "inference_available": False,
                    "latency_seconds": round(latency, 3),
                    "passed": False,
                    "error": repr(exc),
                }

                full_results.append(
                    {
                        "case_id": case_id,
                        "product_ids": [
                            product_1,
                            product_2,
                        ],
                        "query": query,
                        "passed": False,
                        "error": repr(exc),
                    }
                )

            rows.append(row)

    # Save compact CSV summary
    with SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save complete chain outputs
    with FULL_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in full_results:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    passed = sum(row["passed"] for row in rows)

    avg_latency = (
        sum(row["latency_seconds"] for row in rows)
        / len(rows)
    )

    print("\n" + "=" * 60)
    print("PRODUCT COMPARISON EVALUATION")
    print("=" * 60)
    print("Cases:", len(rows))
    print("Passed:", passed)
    print("Failed:", len(rows) - passed)
    print("Pass rate:", round(passed / len(rows), 4))
    print("Average latency:", round(avg_latency, 3), "sec")
    print()
    print("Summary:", SUMMARY_PATH)
    print("Full results:", FULL_PATH)


if __name__ == "__main__":
    main()