"""Offline tests for the section-4 category analytics aggregation functions.

All tests run on small in-memory DataFrames -- no parquet files, no LLM
calls. CategoryAnalyticsChain.run() itself (which does read
products_clean_v1.parquet / comments_clean_v1.parquet) is exercised manually
via the CLI against real data instead; see docs/DECISIONS.md.
"""

from __future__ import annotations

import unittest

import pandas as pd

from src.chains.category_analytics import (
    CategoryAnalyticsNarrative,
    CategoryAnalyticsReport,
    CategoryInsight,
    CategoryScope,
    _numeric_pool,
    _validate_insight_values,
    brand_feedback_comparison,
    category_from_product_ids,
    classify_analytics_question,
    count_no_complaint_mentions,
    dissatisfied_feature_complaints,
    high_volume_low_recommend,
    is_no_complaint,
    resolve_category_from_query,
    suggest_categories,
    top_complaints,
)

COMMENTS = pd.DataFrame(
    [
        {
            "product_id": "1",
            "disadvantages": "قیمت بالا، کیفیت پایین",
            "recommendation_status": "not_recommended",
            "rate": 2.0,
        },
        {
            "product_id": "1",
            "disadvantages": "قیمت بالا",
            "recommendation_status": "recommended",
            "rate": 4.0,
        },
        {
            "product_id": "2",
            "disadvantages": None,
            "recommendation_status": "recommended",
            "rate": 5.0,
        },
        {
            "product_id": "2",
            "disadvantages": "کیفیت پایین",
            "recommendation_status": "not_recommended",
            "rate": 1.0,
        },
        {
            "product_id": "3",
            "disadvantages": "دیر رسید",
            "recommendation_status": "recommended",
            "rate": 3.0,
        },
    ]
)

PRODUCTS = pd.DataFrame(
    [
        {"product_id": "1", "brand": "A"},
        {"product_id": "2", "brand": "A"},
        {"product_id": "3", "brand": "B"},
    ]
)


class TopComplaintsTests(unittest.TestCase):
    def test_counts_recurring_phrases_across_comments(self) -> None:
        table = top_complaints(COMMENTS, top_n=10)
        counts = dict(zip(table["complaint"], table["count"]))
        self.assertEqual(counts["قیمت بالا"], 2)
        self.assertEqual(counts["کیفیت پایین"], 2)
        self.assertEqual(counts["دیر رسید"], 1)

    def test_dissatisfied_feature_complaints_only_uses_not_recommended_rows(self) -> None:
        table = dissatisfied_feature_complaints(COMMENTS, top_n=10)
        counts = dict(zip(table["complaint"], table["count"]))
        self.assertEqual(counts, {"قیمت بالا": 1, "کیفیت پایین": 2})


class NoComplaintPlaceholderTests(unittest.TestCase):
    def test_bare_negations_are_not_complaints(self) -> None:
        for phrase in ("ندارد", "نداره", "هیچی", "ندیدم", "فعلا چیزی ندیدم",
                       "نکته منفی نداره", "ندارد.", "...", "-", ""):
            self.assertTrue(is_no_complaint(phrase), phrase)

    def test_real_complaints_that_start_with_a_negation_survive(self) -> None:
        # The reason the filter matches whole phrases and never prefixes:
        # these are genuine complaints about something missing.
        for phrase in ("نداشتن جعبه", "نداشتن کیف", "نداشتن آداپتور"):
            self.assertFalse(is_no_complaint(phrase), phrase)

    def test_ordinary_complaints_are_kept(self) -> None:
        for phrase in ("قیمت بالا", "کیفیت پایین", "سایز کوچک", "ماندگاری کم"):
            self.assertFalse(is_no_complaint(phrase), phrase)

    def test_placeholders_are_excluded_from_the_complaint_ranking(self) -> None:
        comments = pd.DataFrame(
            [
                {"disadvantages": "ندارد", "recommendation_status": "recommended", "rate": 5.0,
                 "product_id": "1"},
                {"disadvantages": "ندارد", "recommendation_status": "recommended", "rate": 5.0,
                 "product_id": "1"},
                {"disadvantages": "ندارد", "recommendation_status": "recommended", "rate": 5.0,
                 "product_id": "1"},
                {"disadvantages": "قیمت بالا", "recommendation_status": "not_recommended",
                 "rate": 2.0, "product_id": "1"},
            ]
        )
        table = top_complaints(comments, top_n=10)
        self.assertEqual(list(table["complaint"]), ["قیمت بالا"])
        self.assertEqual(count_no_complaint_mentions(comments), 3)


class HighVolumeLowRecommendTests(unittest.TestCase):
    def test_filters_by_min_reviews_and_computes_recommended_rate(self) -> None:
        table = high_volume_low_recommend(COMMENTS, min_reviews=2, top_n=10)
        by_product = table.set_index("product_id")

        self.assertEqual(set(by_product.index), {"1", "2"})  # product 3 has only 1 review
        for product_id in ("1", "2"):
            self.assertEqual(by_product.loc[product_id, "review_count"], 2)
            self.assertAlmostEqual(by_product.loc[product_id, "recommended_rate"], 0.5)


class BrandFeedbackComparisonTests(unittest.TestCase):
    def test_joins_products_and_aggregates_per_brand(self) -> None:
        table = brand_feedback_comparison(COMMENTS, PRODUCTS, min_reviews=1)
        by_brand = table.set_index("brand")

        self.assertEqual(by_brand.loc["A", "review_count"], 4)
        self.assertAlmostEqual(by_brand.loc["A", "recommended_rate"], 0.5)
        self.assertAlmostEqual(by_brand.loc["A", "mean_rate"], 3.0)
        self.assertEqual(by_brand.loc["B", "review_count"], 1)
        self.assertAlmostEqual(by_brand.loc["B", "recommended_rate"], 1.0)

    def test_min_reviews_excludes_thin_brands(self) -> None:
        table = brand_feedback_comparison(COMMENTS, PRODUCTS, min_reviews=2)
        self.assertEqual(set(table["brand"]), {"A"})


class ResolveCategoryFromQueryTests(unittest.TestCase):
    def test_matches_a_known_category_name(self) -> None:
        products = pd.DataFrame({"cat1": ["اسباب بازی", "لباس زنانه"]})
        scope = resolve_category_from_query(
            "پرتکرارترین شکایت در دسته اسباب بازی چیست؟", products
        )
        self.assertEqual(scope, CategoryScope(cat1="اسباب بازی"))

    def test_returns_none_when_nothing_matches(self) -> None:
        products = pd.DataFrame({"cat1": ["اسباب بازی", "لباس زنانه"]})
        self.assertIsNone(resolve_category_from_query("این دسته چطوره؟", products))

    def test_longer_category_name_wins_over_a_shorter_substring(self) -> None:
        products = pd.DataFrame({"cat1": ["کیف", "کیف چرم"]})
        scope = resolve_category_from_query("کیف چرم مشکی رو نشونم بده", products)
        self.assertEqual(scope, CategoryScope(cat1="کیف چرم"))


class AnalyticsQuestionRoutingTests(unittest.TestCase):
    """One answer per question type, not one answer for all four.

    Ali, 2026-08-30. The four questions below are copied verbatim from section
    4 of docs/PROJECT_BRIEF.md. Before this, the question text never reached
    the chain, so all four returned the complaints table.
    """

    def test_the_four_brief_questions_each_pick_their_own_table(self) -> None:
        cases = {
            "پرتکرارترین شکایت کاربران در این دسته چیست؟": "top_complaints",
            "کاربران بیشتر درباره‌ی چه ویژگی‌هایی از محصولات این دسته ناراضی هستند؟":
                "dissatisfied_features",
            "کدام محصولات نظر زیادی دارند اما درصد پیشنهاد خرید آن‌ها پایین است؟":
                "low_recommend_products",
            "بازخورد کاربران درباره‌ی چند برند اصلی این دسته چه تفاوتی دارد؟":
                "brand_feedback",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(classify_analytics_question(query), expected)

    def test_an_unrecognised_question_falls_back_to_overview(self) -> None:
        """Better a category summary than a confidently wrong table."""
        self.assertEqual(
            classify_analytics_question("تحلیل دسته اسباب بازی"), "overview"
        )

    def test_a_brand_question_about_complaints_is_a_brand_question(self) -> None:
        self.assertEqual(
            classify_analytics_question("پرتکرارترین شکایت برندهای این دسته"),
            "brand_feedback",
        )


class CategoryFallbackTests(unittest.TestCase):
    """"این دسته" with no category named -- the brief's own phrasing."""

    @staticmethod
    def _products() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "product_id": ["1", "2", "3", "4"],
                "cat1": ["اسباب بازی", "اسباب بازی", "لباس زنانه", "اسباب بازی"],
                "cat2": [None] * 4,
                "sub_cat": ["toy", "toy", "clothe", "toy"],
                "brand": ["A", "B", "C", "D"],
            }
        )

    def test_the_category_is_taken_from_products_already_in_context(self) -> None:
        scope = category_from_product_ids(self._products(), ["3"])

        self.assertIsNotNone(scope)
        self.assertEqual(scope.cat1, "لباس زنانه")

    def test_one_stray_id_cannot_outvote_the_rest(self) -> None:
        scope = category_from_product_ids(self._products(), ["1", "2", "3"])

        self.assertEqual(scope.cat1, "اسباب بازی")

    def test_unknown_ids_resolve_to_nothing_rather_than_a_guess(self) -> None:
        self.assertIsNone(category_from_product_ids(self._products(), ["9999"]))
        self.assertIsNone(category_from_product_ids(self._products(), []))

    def test_suggestions_are_real_categories_ordered_by_size(self) -> None:
        suggestions = suggest_categories(self._products(), limit=2)

        self.assertEqual(suggestions, ["اسباب بازی", "لباس زنانه"])


class ReportRenderingTests(unittest.TestCase):
    """Each question renders its own table, and the numbers reach the user."""

    @staticmethod
    def _report(question: str, narrative=None) -> CategoryAnalyticsReport:
        return CategoryAnalyticsReport(
            scope=CategoryScope(cat1="اسباب بازی"),
            product_count=2,
            comment_count=7,
            no_complaint_mentions=0,
            top_complaints=pd.DataFrame(
                {"complaint": ["قیمت بالا", "کیفیت پایین"], "count": [836, 453]}
            ),
            dissatisfied_feature_complaints=pd.DataFrame(
                {"complaint": ["بی کیفیت"], "count": [256]}
            ),
            high_volume_low_recommend=pd.DataFrame(
                {
                    "product_id": ["1092337"],
                    "review_count": [65],
                    "recommended_count": [0],
                    "recommended_rate": [0.0],
                }
            ),
            brand_feedback=pd.DataFrame(
                {
                    "brand": ["فکرآذین", "پروت"],
                    "review_count": [36, 30],
                    "recommended_rate": [0.962, 0.0],
                    "mean_rate": [4.29, 1.5],
                }
            ),
            narrative=narrative,
            question=question,
        )

    def test_the_complaints_question_lists_complaints_with_counts(self) -> None:
        rendered = self._report("top_complaints").render_fa()

        self.assertIn("قیمت بالا", rendered)
        self.assertIn("836", rendered)
        self.assertNotIn("فکرآذین", rendered)

    def test_the_brand_question_lists_brands_not_complaints(self) -> None:
        rendered = self._report("brand_feedback").render_fa()

        self.assertIn("فکرآذین", rendered)
        self.assertIn("96.2", rendered)
        self.assertIn("ضعیف‌ترین برند", rendered)
        self.assertNotIn("قیمت بالا", rendered)

    def test_the_low_recommend_question_lists_products(self) -> None:
        rendered = self._report("low_recommend_products").render_fa()

        self.assertIn("1092337", rendered)
        self.assertIn("65", rendered)
        self.assertNotIn("قیمت بالا", rendered)

    def test_the_dissatisfied_question_uses_its_own_table(self) -> None:
        rendered = self._report("dissatisfied_features").render_fa()

        self.assertIn("بی کیفیت", rendered)
        self.assertIn("256", rendered)

    def test_the_narrative_is_appended_and_never_replaces_the_table(self) -> None:
        """It used to be returned alone, hiding every computed number."""
        narrative = CategoryAnalyticsNarrative(
            summary_fa="خلاصه‌ی مدل.", insights=[]
        )
        rendered = self._report("brand_feedback", narrative).render_fa()

        self.assertIn("فکرآذین", rendered)
        self.assertIn("جمع‌بندی مدل:", rendered)
        self.assertIn("خلاصه‌ی مدل.", rendered)

    def test_an_empty_table_says_so_instead_of_crashing(self) -> None:
        report = self._report("brand_feedback")
        empty = CategoryAnalyticsReport(
            scope=report.scope,
            product_count=0,
            comment_count=0,
            no_complaint_mentions=0,
            top_complaints=pd.DataFrame(columns=["complaint", "count"]),
            dissatisfied_feature_complaints=pd.DataFrame(columns=["complaint", "count"]),
            high_volume_low_recommend=pd.DataFrame(
                columns=["product_id", "review_count", "recommended_count", "recommended_rate"]
            ),
            brand_feedback=pd.DataFrame(
                columns=["brand", "review_count", "recommended_rate", "mean_rate"]
            ),
            narrative=None,
            question="brand_feedback",
        )

        self.assertIn("هیچ برندی", empty.render_fa())


class ValidateInsightValuesTests(unittest.TestCase):
    def test_accepts_a_value_copied_from_the_table(self) -> None:
        pool = _numeric_pool(pd.DataFrame({"recommended_rate": [0.5]}))
        narrative = CategoryAnalyticsNarrative(
            summary_fa="خلاصه",
            insights=[CategoryInsight(metric="recommended_rate", value=0.5, text_fa="متن")],
        )
        _validate_insight_values(narrative, pool)  # must not raise

    def test_accepts_the_percentage_form_of_a_table_value(self) -> None:
        pool = _numeric_pool(pd.DataFrame({"recommended_rate": [0.5]}))
        narrative = CategoryAnalyticsNarrative(
            summary_fa="خلاصه",
            insights=[CategoryInsight(metric="recommended_rate_pct", value=50.0, text_fa="متن")],
        )
        _validate_insight_values(narrative, pool)  # must not raise

    def test_rejects_a_fabricated_value(self) -> None:
        pool = _numeric_pool(pd.DataFrame({"recommended_rate": [0.5]}))
        narrative = CategoryAnalyticsNarrative(
            summary_fa="خلاصه",
            insights=[CategoryInsight(metric="made_up", value=99.9, text_fa="متن")],
        )
        with self.assertRaises(ValueError):
            _validate_insight_values(narrative, pool)

    def test_an_object_column_of_floats_still_reaches_the_pool(self) -> None:
        """The exact shape that made a correct insight look fabricated.

        Ali, 2026-08-30. brand_feedback.recommended_rate came out of a groupby
        .apply as an object column, so the old dtype-based selection skipped
        it and the live run rejected 95.569 -- which is 0.955696 x 100, copied
        straight from the table. Every other test here builds a float64 frame,
        which is why none of them saw it.
        """
        table = pd.DataFrame({"recommended_rate": [0.955696]}).astype(object)
        self.assertEqual(table["recommended_rate"].dtype, object)

        pool = _numeric_pool(table)
        narrative = CategoryAnalyticsNarrative(
            summary_fa="خلاصه",
            insights=[
                CategoryInsight(
                    metric="brand_feedback_recommended_rate", value=95.569, text_fa="متن"
                )
            ],
        )
        _validate_insight_values(narrative, pool)  # must not raise

    def test_text_columns_admit_nothing_to_the_pool(self) -> None:
        """Coercing every column must not smuggle in numbers from prose."""
        pool = _numeric_pool(pd.DataFrame({"complaint": ["قیمت بالا", "کیفیت پایین"]}))

        self.assertEqual(pool, [])

    def test_brand_feedback_recommended_rate_is_a_float_column(self) -> None:
        comments = pd.DataFrame(
            {
                "product_id": ["1"] * 3 + ["2"] * 3,
                "recommendation_status": [
                    "recommended", "recommended", "not_recommended",
                    "recommended", "not_recommended", "not_recommended",
                ],
                "rate": [5.0, 4.0, 2.0, 5.0, 1.0, 2.0],
            }
        )
        products = pd.DataFrame({"product_id": ["1", "2"], "brand": ["الف", "ب"]})

        table = brand_feedback_comparison(comments, products, min_reviews=1)

        self.assertEqual(table["recommended_rate"].dtype, "float64")
        self.assertIn("recommended_rate", table.select_dtypes(include="number").columns)


if __name__ == "__main__":
    unittest.main()
