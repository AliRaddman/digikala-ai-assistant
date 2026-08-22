"""Shared retrieval interface for the whole system.

Owner: Ali. Locked signature — chains must depend only on what is defined
here, never on a concrete index. Until the real index is published, everyone
develops against MockRetriever and switches with one config flag.
"""

from __future__ import annotations

import hashlib
import os
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvidenceKind = Literal["product", "comment"]


@dataclass(slots=True)
class Evidence:
    """One retrieved item, either a product or a user comment.

    Both kinds share this shape so a chain can cite them uniformly. `id` is
    the citable identifier the final answer must reference: product_id for a
    product, comment_id for a comment.
    """

    id: str
    kind: EvidenceKind
    text: str
    score: float
    product_id: str | None = None
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        """Short tag the LLM is asked to put next to every claim."""
        prefix = "product" if self.kind == "product" else "comment"
        return f"[{prefix}:{self.id}]"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_prompt_block(self, max_chars: int = 400) -> str:
        """Render for injection into a prompt, truncated to keep tokens down."""
        body = self.text[:max_chars]
        header = self.title or self.product_id or ""
        return f"{self.citation()} {header}\n{body}".strip()


@dataclass(slots=True)
class RetrievalFilters:
    """Structured constraints extracted from the user's Persian query.

    Prices are in Rial, matching the raw data. `rate` is on the 0-100 scale of
    the products table, and is null for unrated products, so min_rate always
    implicitly excludes them.
    """

    price_min: float | None = None
    price_max: float | None = None
    brands: list[str] | None = None
    cat1: list[str] | None = None
    sub_cat: list[str] | None = None
    min_rate: float | None = None
    min_rate_count: int | None = None
    exclude_fake: bool = False
    product_ids: list[str] | None = None

    def is_empty(self) -> bool:
        return all(
            v in (None, False, [], {}) for v in asdict(self).values()
        )


class Retriever(ABC):
    """Every retriever in the project implements exactly this."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:
        """Return at most `top_k` items, sorted by descending score."""


_MOCK_PRODUCTS: list[dict[str, Any]] = [
    {
        "id": "3901234",
        "title": "کوله پشتی روزمره مدل City ظرفیت 20 لیتر",
        "brand": "متفرقه",
        "price": 1_450_000,
        "rate": 86,
        "rate_count": 412,
        "cat1": "کیف و کوله",
        "sub_cat": "clothe",
    },
    {
        "id": "7712045",
        "title": "کیف دوشی چرم طبیعی مدل رها",
        "brand": "چرم مشهد",
        "price": 8_900_000,
        "rate": 91,
        "rate_count": 128,
        "cat1": "کیف زنانه",
        "sub_cat": "clothe",
    },
    {
        "id": "1120876",
        "title": "چمدان مسافرتی چرخ دار سایز متوسط",
        "brand": "پیرکاردین",
        "price": 24_500_000,
        "rate": 78,
        "rate_count": 57,
        "cat1": "لوازم سفر",
        "sub_cat": "travel",
    },
    {
        "id": "6604311",
        "title": "پاور بانک 20000 میلی آمپر ساعت مدل PB20",
        "brand": "انکر",
        "price": 6_200_000,
        "rate": 93,
        "rate_count": 1904,
        "cat1": "لوازم جانبی موبایل",
        "sub_cat": "travel",
    },
    {
        "id": "5583920",
        "title": "کتاب صد سال تنهایی اثر گابریل گارسیا مارکز",
        "brand": "نشر چشمه",
        "price": 3_200_000,
        "rate": 95,
        "rate_count": 803,
        "cat1": "کتاب چاپی",
        "sub_cat": "book & stationary & art",
    },
]

_MOCK_COMMENTS: list[dict[str, Any]] = [
    {
        "id": "51230011",
        "product_id": "3901234",
        "text": "کیفیت دوختش خوبه و برای استفاده روزانه کاملا مناسبه. فقط جیب داخلیش کوچیکه.",
        "rate": 4.0,
        "recommendation_status": "recommended",
        "is_buyer": True,
    },
    {
        "id": "51230044",
        "product_id": "3901234",
        "text": "بعد از دو ماه زیپش خراب شد. به قیمتش نمی ارزه.",
        "rate": 2.0,
        "recommendation_status": "not_recommended",
        "is_buyer": True,
    },
    {
        "id": "51231987",
        "product_id": "3901234",
        "text": "سبکه و جادار. برای مدرسه گرفتم راضیم.",
        "rate": 5.0,
        "recommendation_status": "recommended",
        "is_buyer": True,
    },
    {
        "id": "60019922",
        "product_id": "6604311",
        "text": "شارژدهی واقعیش حدود 14000 هست نه 20000 ولی بازم خوبه.",
        "rate": 4.0,
        "recommendation_status": "recommended",
        "is_buyer": True,
    },
    {
        "id": "60020155",
        "product_id": "6604311",
        "text": "خیلی سنگینه و توی سفر اذیت میکنه.",
        "rate": 3.0,
        "recommendation_status": "no_idea",
        "is_buyer": False,
    },
]


class MockRetriever(Retriever):
    """Fake retriever with fixed, deterministic output.

    Same query always returns the same items, so a chain's output can be
    diffed across runs while the real index does not exist yet. It stays in
    the codebase after the real retriever lands, because it is the only way to
    test a chain without loading a multi-gigabyte index.
    """

    def __init__(self, kind: EvidenceKind = "product") -> None:
        self.kind = kind

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[Evidence]:
        rows = _MOCK_PRODUCTS if self.kind == "product" else _MOCK_COMMENTS
        rows = [r for r in rows if self._passes(r, filters)]

        seed = int(hashlib.md5(query.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        scored = [(round(rng.uniform(0.45, 0.95), 3), r) for r in rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [self._to_evidence(row, score) for score, row in scored[:top_k]]

    def _passes(self, row: dict[str, Any], filters: RetrievalFilters | None) -> bool:
        if filters is None:
            return True
        pid = row.get("product_id", row["id"])
        if filters.product_ids and pid not in filters.product_ids:
            return False
        if self.kind != "product":
            return True
        if filters.price_min is not None and row["price"] < filters.price_min:
            return False
        if filters.price_max is not None and row["price"] > filters.price_max:
            return False
        if filters.brands and row["brand"] not in filters.brands:
            return False
        if filters.cat1 and row["cat1"] not in filters.cat1:
            return False
        if filters.sub_cat and row["sub_cat"] not in filters.sub_cat:
            return False
        if filters.min_rate is not None and row["rate"] < filters.min_rate:
            return False
        if filters.min_rate_count is not None and row["rate_count"] < filters.min_rate_count:
            return False
        return True

    def _to_evidence(self, row: dict[str, Any], score: float) -> Evidence:
        if self.kind == "product":
            return Evidence(
                id=row["id"],
                kind="product",
                text=row["title"],
                score=score,
                product_id=row["id"],
                title=row["title"],
                meta={
                    "brand": row["brand"],
                    "price": row["price"],
                    "rate": row["rate"],
                    "rate_count": row["rate_count"],
                    "cat1": row["cat1"],
                    "sub_cat": row["sub_cat"],
                },
            )
        return Evidence(
            id=row["id"],
            kind="comment",
            text=row["text"],
            score=score,
            product_id=row["product_id"],
            title=None,
            meta={
                "rate": row["rate"],
                "recommendation_status": row["recommendation_status"],
                "is_buyer": row["is_buyer"],
            },
        )


def build_retriever(kind: EvidenceKind = "product", mode: str | None = None) -> Retriever:
    """Return the retriever selected by RETRIEVER_MODE ('mock' or 'real').

    Chains call this and never instantiate a retriever directly, so switching
    the whole system to the real index is a one-line env change.
    """
    mode = (mode or os.getenv("RETRIEVER_MODE", "mock")).lower()
    if mode == "mock":
        return MockRetriever(kind=kind)
    if mode == "real":
        raise NotImplementedError(
            "real retriever is not published yet — keep RETRIEVER_MODE=mock"
        )
    raise ValueError(f"unknown RETRIEVER_MODE: {mode!r}")


if __name__ == "__main__":
    products = build_retriever("product")
    for ev in products.retrieve("یک کیف برای استفاده روزمره که گران نباشد", top_k=3):
        print(ev.score, ev.citation(), ev.title)

    print()
    comments = build_retriever("comment")
    filters = RetrievalFilters(product_ids=["3901234"])
    for ev in comments.retrieve("ایراد این محصول چیست؟", top_k=3, filters=filters):
        print(ev.score, ev.citation(), ev.text)