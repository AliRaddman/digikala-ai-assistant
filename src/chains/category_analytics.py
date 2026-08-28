"""Section 4: category-level analytics for store managers.

Owner: Ali. This is aggregation, not retrieval: it reads
comments_clean_v1.parquet directly (the full, uncapped table -- the comment
retrieval index built for section 2 is scoped to single products and is the
wrong tool here) joined against products_clean_v1.parquet for brand/category
fields. The four functions below are plain pandas; the LLM only turns an
already-computed table into Persian prose. Every number that can appear in
that prose must come from one of the tables -- see _validate_insight_values.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.normalize import normalize
from src.llm.client import CachedLLMClient, build_openai_client

DEFAULT_PRODUCTS_PATH = Path("data/processed/products_clean_v1.parquet")
DEFAULT_COMMENTS_PATH = Path("data/processed/comments_clean_v1.parquet")

PRODUCT_COLUMNS = ["product_id", "cat1", "cat2", "sub_cat", "brand"]
COMMENT_COLUMNS = ["product_id", "disadvantages", "recommendation_status", "rate"]

SUMMARY_PROMPT_VERSION = "category-analytics-summary-v1"

SUMMARY_SYSTEM_PROMPT = """You write a short Persian summary of category-level
shopping analytics for a store category manager.

You are given four already-computed tables: recurring complaints, complaints
from users who did not recommend the product, products with many reviews but
a low recommend rate, and per-brand feedback. These numbers are final and
correct -- you must not recompute, estimate, or invent any number.

Every numeric value you cite in `insights` must be copied exactly from a cell
in the tables you were given, or be that cell's value multiplied by 100 (its
natural percentage form). Do not report a number that does not appear in the
tables in one of those two forms.

summary_fa is a short (2-4 sentence) Persian narrative combining the most
noteworthy patterns across the tables. insights is a structured breakdown:
one entry per notable numeric fact used in summary_fa, with an English
snake_case metric name and its numeric value.
"""

_PHRASE_SPLIT_RE = re.compile(r"[،,؛;\n]+")

# "No disadvantage" placeholders. 20.5% of all disadvantage phrases in the
# cleaned table are one of these -- users write "ندارد" in the cons field to
# mean the product has no downside -- so counting them makes "ندارد" the
# top "complaint" of every category, which answers the brief's question 1
# with a non-answer. Filtering is by whole phrase, never by prefix: real
# complaints like "نداشتن جعبه" and "نداشتن کیف" start the same way and must
# survive. See docs/DECISIONS.md.
_NO_COMPLAINT_FILLERS = {
    "فعلا", "چیزی", "واقعا", "هنوز", "تا", "الان", "حالا",
    "مورد", "موردی", "نکته", "منفی", "خاصی", "که",
}
_NO_COMPLAINT_CORE = {
    "ندارد", "نداره", "ندارم", "ندار", "ندارن", "ندارع", "نداریم",
    "نداشت", "نداشتم", "نداشتیم", "نداشته",
    "ندیدم", "ندیدیم", "ندیده", "نبود", "نبوده", "نیست",
    "هیچ", "هیچی", "هیچکدام",
}
_PHRASE_TRIM = " .-_…؟!\"'()[]"


def is_no_complaint(phrase: str) -> bool:
    """True when a disadvantage phrase is really "there is no disadvantage".

    A phrase qualifies only if every token left after dropping filler words
    is itself a bare negation, so "فعلا چیزی ندیدم" is filtered while
    "نداشتن جعبه" (a genuine packaging complaint) is not.
    """
    tokens = [token.strip(_PHRASE_TRIM) for token in phrase.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return True
    content = [token for token in tokens if token not in _NO_COMPLAINT_FILLERS]
    return bool(content) and all(token in _NO_COMPLAINT_CORE for token in content)


def count_no_complaint_mentions(comments: pd.DataFrame) -> int:
    """How many reviews explicitly stated the product has no downside.

    Reported rather than silently dropped: for a category manager "12% of
    reviewers said there is nothing wrong with it" is itself a finding.
    """
    return sum(
        any(is_no_complaint(phrase) for phrase in _split_phrases(text))
        for text in comments["disadvantages"].dropna()
    )


class CategoryInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    value: float
    text_fa: str = Field(min_length=1)


class CategoryAnalyticsNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_fa: str = Field(min_length=1)
    insights: list[CategoryInsight] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CategoryScope:
    """Which products count as "this category". At least one field is required."""

    cat1: str | None = None
    sub_cat: str | None = None

    def is_empty(self) -> bool:
        return self.cat1 is None and self.sub_cat is None

    def label_fa(self) -> str:
        return self.sub_cat or self.cat1 or "این دسته"


@dataclass(frozen=True, slots=True)
class CategoryAnalyticsReport:
    scope: CategoryScope
    product_count: int
    comment_count: int
    no_complaint_mentions: int
    top_complaints: pd.DataFrame
    dissatisfied_feature_complaints: pd.DataFrame
    high_volume_low_recommend: pd.DataFrame
    brand_feedback: pd.DataFrame
    narrative: CategoryAnalyticsNarrative | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": {"cat1": self.scope.cat1, "sub_cat": self.scope.sub_cat},
            "product_count": self.product_count,
            "comment_count": self.comment_count,
            "no_complaint_mentions": self.no_complaint_mentions,
            "top_complaints": _records(self.top_complaints),
            "dissatisfied_feature_complaints": _records(self.dissatisfied_feature_complaints),
            "high_volume_low_recommend": _records(self.high_volume_low_recommend),
            "brand_feedback": _records(self.brand_feedback),
            "narrative": self.narrative.model_dump(mode="json") if self.narrative else None,
        }

    def render_fa(self) -> str:
        if self.narrative:
            return self.narrative.summary_fa
        lines = [
            f"تحلیل دسته «{self.scope.label_fa()}» بر پایه‌ی {self.comment_count} نظر "
            f"از {self.product_count} محصول:"
        ]
        if not self.top_complaints.empty:
            top = self.top_complaints.iloc[0]
            lines.append(f"پرتکرارترین شکایت: «{top['complaint']}» ({int(top['count'])} بار)")
        else:
            lines.append("نظری برای این دسته پیدا نشد.")
        return "\n".join(lines)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame rows as plain JSON-safe python values (no numpy scalars)."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _split_phrases(text: object) -> list[str]:
    """A comment's disadvantages field into individual complaint phrases."""
    if not isinstance(text, str) or not text.strip():
        return []
    return [normalize(part) for part in _PHRASE_SPLIT_RE.split(text) if normalize(part)]


@lru_cache(maxsize=1)
def load_products(path: str = str(DEFAULT_PRODUCTS_PATH)) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=PRODUCT_COLUMNS)
    df["product_id"] = df["product_id"].astype(str)
    return df


def resolve_category_from_query(query: str, products: pd.DataFrame) -> CategoryScope | None:
    """Zero-cost rule-based match: the longest known cat1 value that appears
    verbatim in the normalized query wins, so a specific category name is not
    shadowed by a shorter, coincidentally-matching substring."""
    normalized_query = normalize(query)
    known_cat1 = sorted(
        (value for value in products["cat1"].dropna().unique() if value),
        key=len,
        reverse=True,
    )
    for value in known_cat1:
        if value in normalized_query:
            return CategoryScope(cat1=value)
    return None


def category_product_ids(products: pd.DataFrame, scope: CategoryScope) -> list[str]:
    mask = pd.Series(True, index=products.index)
    if scope.cat1:
        mask &= products["cat1"] == scope.cat1
    if scope.sub_cat:
        mask &= products["sub_cat"] == scope.sub_cat
    return products.loc[mask, "product_id"].tolist()


def load_category_comments(comments_path: Path, product_ids: list[str]) -> pd.DataFrame:
    """Read only the rows and columns this chain needs.

    comments_clean_v1.parquet is several million rows; pyarrow's row-level
    `filters` push the product_id membership check into the scan itself, so
    only this category's rows are ever materialized in memory.
    """
    if not product_ids:
        return pd.DataFrame(columns=COMMENT_COLUMNS)
    return pd.read_parquet(
        comments_path,
        columns=COMMENT_COLUMNS,
        filters=[("product_id", "in", product_ids)],
    )


def top_complaints(comments: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Most frequent real disadvantage phrases (question 1: پرتکرارترین شکایت).

    "No disadvantage" placeholders are excluded here -- see is_no_complaint --
    and counted separately by count_no_complaint_mentions.
    """
    counter: Counter[str] = Counter()
    for text in comments["disadvantages"].dropna():
        counter.update(
            phrase for phrase in _split_phrases(text) if not is_no_complaint(phrase)
        )
    return pd.DataFrame(counter.most_common(top_n), columns=["complaint", "count"])


def dissatisfied_feature_complaints(comments: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Same as top_complaints, restricted to not_recommended reviews
    (question 2: چه ویژگی‌هایی کاربران را ناراضی کرده)."""
    subset = comments.loc[comments["recommendation_status"] == "not_recommended"]
    return top_complaints(subset, top_n=top_n)


def high_volume_low_recommend(
    comments: pd.DataFrame, min_reviews: int = 20, top_n: int = 10
) -> pd.DataFrame:
    """Products with many reviews but a low recommend rate (question 3)."""
    grouped = comments.groupby("product_id")["recommendation_status"]
    review_count = grouped.size()
    recommended_count = grouped.apply(lambda s: (s == "recommended").sum())
    table = pd.DataFrame(
        {
            "review_count": review_count,
            "recommended_count": recommended_count,
            "recommended_rate": recommended_count / review_count,
        }
    )
    table = table.loc[table["review_count"] >= min_reviews].sort_values(
        "recommended_rate", ascending=True
    )
    return table.head(top_n).reset_index()


def brand_feedback_comparison(
    comments: pd.DataFrame, products: pd.DataFrame, min_reviews: int = 30
) -> pd.DataFrame:
    """Recommend rate and average rating per major brand (question 4)."""
    merged = comments.merge(products[["product_id", "brand"]], on="product_id", how="left")
    grouped = merged.groupby("brand")
    review_count = grouped.size()
    table = pd.DataFrame(
        {
            "review_count": review_count,
            "recommended_rate": grouped["recommendation_status"].apply(
                lambda s: (s == "recommended").mean()
            ),
            "mean_rate": grouped["rate"].mean(),
        }
    )
    table = table.loc[table["review_count"] >= min_reviews].sort_values(
        "review_count", ascending=False
    )
    return table.reset_index()


def _numeric_pool(*tables: pd.DataFrame) -> list[float]:
    pool: list[float] = []
    for table in tables:
        for column in table.select_dtypes(include="number").columns:
            for value in table[column].dropna():
                value = float(value)
                pool.append(value)
                pool.append(round(value * 100, 1))
    return pool


def _validate_insight_values(
    narrative: CategoryAnalyticsNarrative, pool: list[float], tolerance: float = 0.05
) -> None:
    unverified = [
        insight
        for insight in narrative.insights
        if not any(abs(insight.value - candidate) <= tolerance for candidate in pool)
    ]
    if unverified:
        raise ValueError(
            "narrative cites values absent from the aggregated tables: "
            + ", ".join(f"{item.metric}={item.value}" for item in unverified)
        )


class CategoryAnalyticsChain:
    def __init__(
        self,
        *,
        products_path: Path = DEFAULT_PRODUCTS_PATH,
        comments_path: Path = DEFAULT_COMMENTS_PATH,
        client: CachedLLMClient | None = None,
    ) -> None:
        self.products_path = products_path
        self.comments_path = comments_path
        self.client = client

    def run(
        self,
        scope: CategoryScope,
        *,
        min_reviews_for_ranking: int = 20,
        top_n: int = 10,
    ) -> CategoryAnalyticsReport:
        if scope.is_empty():
            raise ValueError("scope must specify at least cat1 or sub_cat")

        products = load_products(str(self.products_path))
        product_ids = category_product_ids(products, scope)
        comments = load_category_comments(self.comments_path, product_ids)

        top_c = top_complaints(comments, top_n=top_n)
        dissatisfied = dissatisfied_feature_complaints(comments, top_n=top_n)
        low_recommend = high_volume_low_recommend(
            comments, min_reviews=min_reviews_for_ranking, top_n=top_n
        )
        brands = brand_feedback_comparison(comments, products, min_reviews=min_reviews_for_ranking)

        narrative = None
        if self.client is not None and not comments.empty:
            narrative = self._summarize(
                scope, len(product_ids), len(comments), top_c, dissatisfied, low_recommend, brands
            )

        return CategoryAnalyticsReport(
            scope=scope,
            product_count=len(product_ids),
            comment_count=len(comments),
            no_complaint_mentions=count_no_complaint_mentions(comments),
            top_complaints=top_c,
            dissatisfied_feature_complaints=dissatisfied,
            high_volume_low_recommend=low_recommend,
            brand_feedback=brands,
            narrative=narrative,
        )

    def _summarize(
        self,
        scope: CategoryScope,
        product_count: int,
        comment_count: int,
        top_c: pd.DataFrame,
        dissatisfied: pd.DataFrame,
        low_recommend: pd.DataFrame,
        brands: pd.DataFrame,
    ) -> CategoryAnalyticsNarrative:
        assert self.client is not None
        payload = {
            "category": scope.label_fa(),
            "product_count": product_count,
            "comment_count": comment_count,
            "top_complaints": _records(top_c),
            "dissatisfied_feature_complaints": _records(dissatisfied),
            "high_volume_low_recommend": _records(low_recommend),
            "brand_feedback": _records(brands),
        }
        result = self.client.generate_structured(
            operation="summarize_category_analytics",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            response_model=CategoryAnalyticsNarrative,
            cache_namespace=SUMMARY_PROMPT_VERSION,
        )
        narrative = CategoryAnalyticsNarrative.model_validate(result.data)
        _validate_insight_values(narrative, _numeric_pool(top_c, dissatisfied, low_recommend, brands))
        return narrative


def main() -> None:
    parser = argparse.ArgumentParser(description="Category-level review analytics.")
    parser.add_argument("--cat1")
    parser.add_argument("--sub-cat")
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS_PATH)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS_PATH)
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    chain = CategoryAnalyticsChain(
        products_path=args.products,
        comments_path=args.comments,
        client=None if args.no_summary else build_openai_client(),
    )
    report = chain.run(CategoryScope(cat1=args.cat1, sub_cat=args.sub_cat))
    print(report.render_fa())


if __name__ == "__main__":
    main()
