import pandas as pd

from src.data.normalize import normalize
from src.data.comments_cleaning import clean_comments_dataframe


def _sample_comments():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title": "خوبه",
                "body": "می\u200cخوام دوباره بخرم",
                "created_at": "1402/01/01",
                "rate": 5.0,
                "recommendation_status": "no_idea",
                "is_buyer": True,
                "product_id": 100,
                "advantages": "کیفیت خوب",
                "disadvantages": "قیمت بالا",
                "likes": 1,
                "dislikes": 0,
                "seller_title": "seller",
                "seller_code": "A",
                "true_to_size_rate": pd.NA,
            },
            {
                "id": 2,
                "title": pd.NA,
                "body": "بد نبود",
                "created_at": "1402/01/02",
                "rate": 3.0,
                "recommendation_status": pd.NA,
                "is_buyer": True,
                "product_id": 101,
                "advantages": pd.NA,
                "disadvantages": "بسته\u200cبندی ضعیف",
                "likes": 0,
                "dislikes": 0,
                "seller_title": "seller",
                "seller_code": "B",
                "true_to_size_rate": pd.NA,
            },
            # Duplicate comment ID: must be removed.
            {
                "id": 2,
                "title": "duplicate",
                "body": "duplicate",
                "created_at": "1402/01/03",
                "rate": 1.0,
                "recommendation_status": "not_recommended",
                "is_buyer": True,
                "product_id": 101,
                "advantages": "",
                "disadvantages": "",
                "likes": 0,
                "dislikes": 0,
                "seller_title": "seller",
                "seller_code": "B",
                "true_to_size_rate": pd.NA,
            },
            # No body, but useful disadvantage: must be kept.
            {
                "id": 3,
                "title": pd.NA,
                "body": pd.NA,
                "created_at": "1402/01/04",
                "rate": 1.0,
                "recommendation_status": "not_recommended",
                "is_buyer": True,
                "product_id": 102,
                "advantages": pd.NA,
                "disadvantages": "خیلی گران",
                "likes": 0,
                "dislikes": 0,
                "seller_title": "seller",
                "seller_code": "C",
                "true_to_size_rate": pd.NA,
            },
            # Completely empty text: must be removed.
            {
                "id": 4,
                "title": pd.NA,
                "body": pd.NA,
                "created_at": "1402/01/05",
                "rate": 0.0,
                "recommendation_status": "recommended",
                "is_buyer": False,
                "product_id": 103,
                "advantages": pd.NA,
                "disadvantages": pd.NA,
                "likes": 0,
                "dislikes": 0,
                "seller_title": "seller",
                "seller_code": "D",
                "true_to_size_rate": pd.NA,
            },
        ]
    )


def test_three_class_target_and_missing_are_not_mixed():
    cleaned, _ = clean_comments_dataframe(_sample_comments())

    row_no_idea = cleaned.loc[cleaned["id"] == 1].iloc[0]
    row_missing = cleaned.loc[cleaned["id"] == 2].iloc[0]

    assert row_no_idea["recommendation_status"] == "no_idea"
    assert pd.isna(row_missing["recommendation_status"])


def test_shared_normalizer_is_used():
    cleaned, _ = clean_comments_dataframe(_sample_comments())

    actual = cleaned.loc[cleaned["id"] == 1, "body"].iloc[0]
    expected = normalize("می\u200cخوام دوباره بخرم")

    assert actual == expected


def test_advantages_and_disadvantages_stay_separate():
    cleaned, _ = clean_comments_dataframe(_sample_comments())

    row = cleaned.loc[cleaned["id"] == 1].iloc[0]

    assert row["advantages"] == normalize("کیفیت خوب")
    assert row["disadvantages"] == normalize("قیمت بالا")

    # Combined text exists only as an additional representation.
    assert row["advantages"] in row["text"]
    assert row["disadvantages"] in row["text"]


def test_duplicate_ids_are_removed():
    cleaned, report = clean_comments_dataframe(_sample_comments())

    assert cleaned["id"].is_unique
    assert report["duplicate_comment_ids_removed"] == 1


def test_empty_body_with_useful_disadvantage_is_kept():
    cleaned, _ = clean_comments_dataframe(_sample_comments())

    assert 3 in cleaned["id"].tolist()
    row = cleaned.loc[cleaned["id"] == 3].iloc[0]
    assert row["disadvantages"] == normalize("خیلی گران")


def test_completely_textless_row_is_removed():
    cleaned, report = clean_comments_dataframe(_sample_comments())

    assert 4 not in cleaned["id"].tolist()
    assert report["rows_removed_no_text_anywhere"] == 1
