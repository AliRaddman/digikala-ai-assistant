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
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.normalize import normalize, to_search_text
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
    question: AnalyticsQuestion = "overview"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": {"cat1": self.scope.cat1, "sub_cat": self.scope.sub_cat},
            "question": self.question,
            "product_count": self.product_count,
            "comment_count": self.comment_count,
            "no_complaint_mentions": self.no_complaint_mentions,
            "top_complaints": _records(self.top_complaints),
            "dissatisfied_feature_complaints": _records(self.dissatisfied_feature_complaints),
            "high_volume_low_recommend": _records(self.high_volume_low_recommend),
            "brand_feedback": _records(self.brand_feedback),
            "narrative": self.narrative.model_dump(mode="json") if self.narrative else None,
        }

    def render_fa(self, top_n: int = 5) -> str:
        """The table the question asked for, then the model's summary if any.

        Ali, 2026-08-30. This used to return `narrative.summary_fa` alone when
        a narrative existed and the single top complaint otherwise, so all four
        section-4 question types produced the same answer and three of the four
        computed tables never reached the user at all. The facts now come
        first and the model's prose is appended under its own heading, keeping
        the facts/inference split the brief asks for in section 3.
        """
        header = (
            f"تحلیل دسته «{self.scope.label_fa()}» بر پایه‌ی {self.comment_count:,} نظر "
            f"از {self.product_count:,} محصول:"
        )
        lines = [header, ""]
        lines.extend(self._question_lines(top_n))
        if self.narrative:
            lines += ["", "جمع‌بندی مدل:", self.narrative.summary_fa]
        return "\n".join(lines).strip()

    def _question_lines(self, top_n: int) -> list[str]:
        if self.question == "brand_feedback":
            return _render_brand_feedback(self.brand_feedback, top_n)
        if self.question == "low_recommend_products":
            return _render_low_recommend(self.high_volume_low_recommend, top_n)
        if self.question == "dissatisfied_features":
            return _render_complaints(
                self.dissatisfied_feature_complaints,
                top_n,
                "ویژگی‌هایی که ناراضی‌ها بیشتر به آن‌ها اشاره کرده‌اند:",
            )
        if self.question == "top_complaints":
            return _render_complaints(
                self.top_complaints, top_n, "پرتکرارترین شکایت‌ها:"
            )
        return _render_overview(self.top_complaints, self.brand_feedback, top_n)


def _render_complaints(table: pd.DataFrame, top_n: int, heading: str) -> list[str]:
    if table.empty:
        return ["شکایت ثبت‌شده‌ای برای این دسته پیدا نشد."]
    lines = [heading]
    for _, row in table.head(top_n).iterrows():
        lines.append(f"- «{row['complaint']}» — {int(row['count']):,} بار")
    return lines


def _render_low_recommend(table: pd.DataFrame, top_n: int) -> list[str]:
    if table.empty:
        return [
            "هیچ محصولی در این دسته به آستانه‌ی حداقل تعداد نظر برای این رتبه‌بندی نرسید."
        ]
    lines = ["محصولاتی که نظر زیادی دارند ولی نرخ پیشنهاد خریدشان پایین است:"]
    for _, row in table.head(top_n).iterrows():
        rate = row["recommended_rate"]
        rate_fa = "نامشخص" if pd.isna(rate) else f"{float(rate) * 100:.1f}٪"
        lines.append(
            f"- محصول {row['product_id']} — {int(row['review_count']):,} نظر — "
            f"نرخ پیشنهاد {rate_fa}"
        )
    return lines


def _render_brand_feedback(table: pd.DataFrame, top_n: int) -> list[str]:
    if table.empty:
        return ["هیچ برندی در این دسته به آستانه‌ی حداقل تعداد نظر نرسید."]
    ranked = table.sort_values("recommended_rate", ascending=False)
    lines = ["بازخورد کاربران به تفکیک برند (مرتب بر اساس نرخ پیشنهاد):"]
    for _, row in ranked.head(top_n).iterrows():
        rate = row["recommended_rate"]
        rate_fa = "نامشخص" if pd.isna(rate) else f"{float(rate) * 100:.1f}٪"
        mean_rate = row["mean_rate"]
        mean_fa = "—" if pd.isna(mean_rate) else f"{float(mean_rate):.2f} از ۵"
        lines.append(
            f"- {row['brand']} — {int(row['review_count']):,} نظر — "
            f"نرخ پیشنهاد {rate_fa} — میانگین امتیاز {mean_fa}"
        )
    weakest = ranked.iloc[-1]
    weakest_rate = weakest["recommended_rate"]
    if not pd.isna(weakest_rate) and len(ranked) > 1:
        lines.append(
            f"ضعیف‌ترین برند این دسته: {weakest['brand']} با نرخ پیشنهاد "
            f"{float(weakest_rate) * 100:.1f}٪."
        )
    return lines


def _render_overview(
    complaints: pd.DataFrame, brands: pd.DataFrame, top_n: int
) -> list[str]:
    lines = _render_complaints(complaints, min(top_n, 3), "پرتکرارترین شکایت‌ها:")
    if not brands.empty:
        best = brands.sort_values("recommended_rate", ascending=False).iloc[0]
        rate = best["recommended_rate"]
        if not pd.isna(rate):
            lines.append(
                f"بهترین برند از نظر نرخ پیشنهاد: {best['brand']} "
                f"({float(rate) * 100:.1f}٪)."
            )
    return lines


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


AnalyticsQuestion = Literal[
    "top_complaints",
    "dissatisfied_features",
    "low_recommend_products",
    "brand_feedback",
    "overview",
]

# Section 4 of docs/PROJECT_BRIEF.md names four analytical questions. Each maps
# to one already-computed table; the signals below are the phrases those four
# questions and their natural paraphrases actually contain.
#
# Rule-based rather than a model call: it is free, deterministic, testable, and
# the whole set is four classes over a closed vocabulary -- exactly the shape
# where a regex beats an LLM (see failure 8 in docs/FAILURES.md, where the
# model lost to the rule-based filter extractor on the same kind of task).
#
# Ordered most-specific first. A question naming both a brand and a complaint
# is a brand question about complaints, and the brand table is the one that
# cannot be reconstructed from the complaint table.
_QUESTION_SIGNALS: tuple[tuple[AnalyticsQuestion, tuple[str, ...]], ...] = (
    (
        "low_recommend_products",
        ("پیشنهاد خرید", "درصد پیشنهاد", "نرخ پیشنهاد", "نظر زیادی", "پرنظر",
         "نظر زیاد", "توصیه پایین"),
    ),
    ("brand_feedback", ("برند",)),
    (
        "dissatisfied_features",
        ("ویژگی", "ناراضی", "نارضایتی", "رضایت ندارند"),
    ),
    (
        "top_complaints",
        ("شکایت", "مشکل", "ایراد", "انتقاد", "نقطه ضعف", "نقاط ضعف"),
    ),
)


def classify_analytics_question(query: str) -> AnalyticsQuestion:
    """Which of the four section-4 tables this question is asking for.

    Falls back to "overview" rather than guessing, so an unrecognised question
    gets the category summary instead of a confidently wrong table.
    """
    text = f" {to_search_text(query)} "
    for question, signals in _QUESTION_SIGNALS:
        if any(signal in text for signal in signals):
            return question
    return "overview"


def category_from_product_ids(
    products: pd.DataFrame,
    product_ids: list[str],
) -> CategoryScope | None:
    """The category of the products already in the conversation.

    All four example questions in the brief say "این دسته" without naming one,
    which is only answerable if the session already has products in context.
    The most common cat1 among them wins, so one stray id cannot redirect the
    whole analysis.
    """
    if not product_ids:
        return None
    wanted = {str(product_id) for product_id in product_ids}
    matched = products.loc[products["product_id"].isin(wanted), "cat1"].dropna()
    if matched.empty:
        return None
    return CategoryScope(cat1=str(matched.mode().iloc[0]))


def suggest_categories(products: pd.DataFrame, limit: int = 6) -> list[str]:
    """The largest real cat1 values, to offer as a concrete choice."""
    counts = products["cat1"].dropna().value_counts()
    return [str(value) for value in counts.head(limit).index]


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
            # .apply over a str/arrow-backed column returns an object column
            # even though nearly every element is a float. That is invisible
            # here but not in _numeric_pool, which selected by dtype -- see
            # the comment there. to_numeric rather than astype("float64"):
            # a brand whose reviews all lack a recommendation_status yields
            # pd.NA, which astype cannot convert, and NaN is the honest value
            # for a rate that is undefined rather than zero.
            "recommended_rate": pd.to_numeric(
                grouped["recommendation_status"].apply(
                    lambda s: (s == "recommended").mean()
                ),
                errors="coerce",
            ),
            "mean_rate": grouped["rate"].mean(),
        }
    )
    table = table.loc[table["review_count"] >= min_reviews].sort_values(
        "review_count", ascending=False
    )
    return table.reset_index()


def _numeric_pool(*tables: pd.DataFrame) -> list[float]:
    """Every number the model is allowed to cite, in both admissible forms.

    Ali, 2026-08-30. This used to select columns by dtype. brand_feedback's
    recommended_rate was an object column of plain floats, so none of its
    values reached the pool and _validate_insight_values rejected an insight
    that was copied correctly out of the table it was given:

        narrative cites values absent from the aggregated tables:
        brand_feedback_di_ve_joochi_recommended_rate=95.569

    0.955696 is in the table, and SUMMARY_SYSTEM_PROMPT explicitly allows the
    x100 percentage form. The prompt promised something the validator did not
    honour -- the same shape as the rubric/validator contradiction in
    src/eval/grounding.py, and again only a live call could show it.

    The dtype is now fixed at the source too, but coercing here rather than
    trusting dtype means the next such column cannot re-open this hole. Text
    columns coerce to NaN and drop out, so nothing extra is admitted.
    """
    pool: list[float] = []
    for table in tables:
        for column in table.columns:
            for value in pd.to_numeric(table[column], errors="coerce").dropna():
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
        question: AnalyticsQuestion = "overview",
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
            question=question,
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
    parser.add_argument(
        "--question",
        default="overview",
        choices=[
            "overview",
            "top_complaints",
            "dissatisfied_features",
            "low_recommend_products",
            "brand_feedback",
        ],
        help="which section-4 table to render; --ask classifies this from Persian text",
    )
    parser.add_argument("--ask", help="a Persian question; overrides --question")
    args = parser.parse_args()

    chain = CategoryAnalyticsChain(
        products_path=args.products,
        comments_path=args.comments,
        client=None if args.no_summary else build_openai_client(),
    )
    question = classify_analytics_question(args.ask) if args.ask else args.question
    report = chain.run(
        CategoryScope(cat1=args.cat1, sub_cat=args.sub_cat), question=question
    )
    print(report.render_fa())


if __name__ == "__main__":
    main()
