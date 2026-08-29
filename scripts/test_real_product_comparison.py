from pathlib import Path
import json

import pandas as pd

from src.chains.product_comparison import ProductComparisonChain
from src.retrieval.base import Evidence, RetrievalFilters, Retriever
from src.retrieval.comments import CommentRetriever


PRODUCT_META = Path("data/indexes/products_meta_v1.parquet")
COMMENT_INDEX = Path("data/indexes/comparison_eval")
OUTPUT = Path("outputs/product_comparison/real_pair_81934_130874.json")


def clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


class MetadataProductRetriever(Retriever):
    """
    Lightweight evaluation retriever.

    Product IDs are already resolved by the upstream orchestrator,
    so product facts are retrieved directly from the shared metadata table.
    """

    def __init__(self, meta_path: Path):
        self.df = pd.read_parquet(meta_path)
        self.df["product_id"] = self.df["product_id"].astype(str)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:

        df = self.df

        if filters is not None and filters.product_ids:
            product_ids = {
                str(product_id)
                for product_id in filters.product_ids
            }

            df = df[
                df["product_id"].isin(product_ids)
            ]

        results = []

        for _, row in df.head(top_k).iterrows():
            product_id = str(row["product_id"])
            title = str(row["title"])

            results.append(
                Evidence(
                    id=product_id,
                    kind="product",
                    text=title,
                    score=1.0,
                    product_id=product_id,
                    title=title,
                    meta={
                        "brand": clean_value(row.get("brand")),
                        "price": clean_value(row.get("price")),
                        "rate": clean_value(row.get("rate")),
                        "rate_count": clean_value(row.get("rate_count")),
                        "cat1": clean_value(row.get("cat1")),
                        "sub_cat": clean_value(row.get("sub_cat")),
                        "is_fake": clean_value(row.get("is_fake")),
                    },
                )
            )

        return results


def main():

    product_retriever = MetadataProductRetriever(
        PRODUCT_META
    )

    comment_retriever = CommentRetriever(
        index_dir=COMMENT_INDEX
    )

    chain = ProductComparisonChain(
        product_retriever=product_retriever,
        comment_retriever=comment_retriever,
        llm_client=None,
        comment_top_k=5,
    )

    result = chain.run(
        "این دو عطر مردانه را از نظر کیفیت، ماندگاری، رایحه و ارزش خرید مقایسه کن",
        product_ids=[
            "81934",
            "130874",
        ],
    )

    print(result.render_fa())

    print("\n" + "=" * 70)
    print("RETRIEVED REVIEW EVIDENCE")
    print("=" * 70)

    for group in result.evidence:
        print(f"\nPRODUCT: {group.product_id}")

        for comment in group.comments:
            print(
                f"{comment.score:.4f} "
                f"{comment.citation()} "
                f"{comment.text}"
            )

    assert len(result.facts) == 2
    assert len(result.missing_products) == 0
    assert result.as_dict()["evidence"]["available"] is True
    assert result.inference is None

    for group in result.evidence:
        assert all(
            str(comment.product_id) == group.product_id
            for comment in group.comments
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result.as_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n✅ REAL Product Comparison test passed.")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()