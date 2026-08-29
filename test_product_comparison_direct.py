from src.chains.product_comparison import ProductComparisonChain
from src.retrieval.base import build_retriever


def main():
    chain = ProductComparisonChain(
        product_retriever=build_retriever("product", mode="mock"),
        comment_retriever=build_retriever("comment", mode="mock"),
        llm_client=None,
        comment_top_k=5,
    )

    result = chain.run(
        "این دو محصول را از نظر کیفیت و ارزش خرید مقایسه کن",
        product_ids=["3901234", "6604311"],
    )

    print(result.render_fa())
    print("\n--- RAW OUTPUT ---")
    print(result.as_dict())

    assert len(result.facts) == 2
    assert result.as_dict()["evidence"]["available"] is True
    assert result.inference is None
    assert len(result.missing_products) == 0

    print("\n✅ Product Comparison mock test passed.")


if __name__ == "__main__":
    main()
