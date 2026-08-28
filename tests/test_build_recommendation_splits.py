import pandas as pd

from src.data.build_recommendation_splits import build_group_splits


def _sample_df():
    rows = []
    labels = ["recommended", "not_recommended", "no_idea"]

    comment_id = 1

    # 30 products, 6 comments each.
    # Enough groups so all three classes appear in both train and test.
    for product_id in range(100, 130):
        for j in range(6):
            label = labels[(product_id + j) % 3]

            rows.append(
                {
                    "id": comment_id,
                    "product_id": product_id,
                    "title": "",
                    "body": f"review {comment_id}",
                    "advantages": "",
                    "disadvantages": "",
                    "recommendation_status": label,
                    "text": f"review {comment_id}",
                }
            )
            comment_id += 1

    # Missing labels must be excluded from classifier splits, not converted
    # into no_idea.
    for product_id in range(200, 203):
        rows.append(
            {
                "id": comment_id,
                "product_id": product_id,
                "title": "",
                "body": "unlabeled review",
                "advantages": "",
                "disadvantages": "",
                "recommendation_status": pd.NA,
                "text": "unlabeled review",
            }
        )
        comment_id += 1

    return pd.DataFrame(rows)


def test_missing_labels_are_excluded_not_converted():
    df = _sample_df()

    train_df, test_df, report = build_group_splits(
        df,
        test_size=0.2,
        random_state=42,
    )

    combined = pd.concat([train_df, test_df], ignore_index=True)

    assert combined["recommendation_status"].isna().sum() == 0
    assert report["rows_excluded_missing_or_invalid_label"] == 3

    # The real no_idea class must still exist.
    assert "no_idea" in set(
        combined["recommendation_status"].astype(str).unique()
    )


def test_product_ids_do_not_overlap():
    df = _sample_df()

    train_df, test_df, report = build_group_splits(
        df,
        test_size=0.2,
        random_state=42,
    )

    train_products = set(train_df["product_id"])
    test_products = set(test_df["product_id"])

    assert train_products.isdisjoint(test_products)
    assert report["product_overlap_count"] == 0


def test_all_three_classes_exist_in_both_splits():
    df = _sample_df()

    train_df, test_df, _ = build_group_splits(
        df,
        test_size=0.2,
        random_state=42,
    )

    expected = {
        "recommended",
        "not_recommended",
        "no_idea",
    }

    assert set(train_df["recommendation_status"].astype(str).unique()) == expected
    assert set(test_df["recommendation_status"].astype(str).unique()) == expected


def test_split_is_reproducible():
    df = _sample_df()

    train_a, test_a, _ = build_group_splits(
        df,
        test_size=0.2,
        random_state=42,
    )
    train_b, test_b, _ = build_group_splits(
        df,
        test_size=0.2,
        random_state=42,
    )

    assert train_a["id"].tolist() == train_b["id"].tolist()
    assert test_a["id"].tolist() == test_b["id"].tolist()
