"""Section 1: product discovery over the shared Retriever contract.

Owner: Benyamin. The chain works with MockRetriever today and the real
retriever later without changing this file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from src.chains.product_filters import (
    FilterExtractor,
    LLMFilterExtractor,
    ProductFilterPlan,
    RuleBasedFilterExtractor,
)
from src.llm.client import build_openai_client
from src.retrieval.base import Evidence, Retriever, build_retriever


@dataclass(frozen=True, slots=True)
class ProductDiscoveryResult:
    user_query: str
    filter_plan: ProductFilterPlan
    products: list[Evidence]

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "filter_plan": self.filter_plan.model_dump(mode="json"),
            "products": [product.as_dict() for product in self.products],
        }

    def render_fa(self) -> str:
        if not self.products:
            return "محصولی مطابق محدودیت‌های درخواست پیدا نشد."
        lines = ["محصول‌های پیشنهادی:"]
        for index, product in enumerate(self.products, start=1):
            price = product.meta.get("price")
            price_text = (
                f"{price / 10:,.0f} تومان" if isinstance(price, (int, float)) else "نامشخص"
            )
            rate = product.meta.get("rate")
            rate_text = f"{rate}/100" if rate is not None else "بدون امتیاز"
            lines.append(
                f"{index}. {product.title or product.text} — {price_text} — "
                f"امتیاز {rate_text} {product.citation()}"
            )
        return "\n".join(lines)


class ProductDiscoveryChain:
    def __init__(self, retriever: Retriever, extractor: FilterExtractor) -> None:
        self.retriever = retriever
        self.extractor = extractor

    def run(self, user_query: str, top_k: int = 5) -> ProductDiscoveryResult:
        if not user_query.strip():
            raise ValueError("user_query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        plan = self.extractor.extract(user_query)
        products = self.retriever.retrieve(
            plan.search_query,
            top_k=top_k,
            filters=plan.to_retrieval_filters(),
        )
        return ProductDiscoveryResult(
            user_query=user_query,
            filter_plan=plan,
            products=products,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover products from Persian text.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever-mode", default="mock")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the cached OpenAI filter extractor instead of the free baseline.",
    )
    args = parser.parse_args()

    extractor: FilterExtractor
    if args.use_llm:
        extractor = LLMFilterExtractor(build_openai_client())
    else:
        extractor = RuleBasedFilterExtractor()
    chain = ProductDiscoveryChain(
        retriever=build_retriever("product", mode=args.retriever_mode),
        extractor=extractor,
    )
    print(chain.run(args.query, top_k=args.top_k).render_fa())


if __name__ == "__main__":
    main()
