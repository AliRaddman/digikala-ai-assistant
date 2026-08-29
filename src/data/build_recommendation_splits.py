"""
Build leakage-free recommendation-status train/test splits.

Input
-----
data/processed/comments_processed.parquet

Outputs
-------
data/processed/train_group_split.parquet
data/processed/test_group_split.parquet
data/processed/recommendation_split_report.json

Project rules
-------------
- Keep exactly three target classes:
    recommended
    not_recommended
    no_idea
- Missing recommendation_status is NOT the same as no_idea and is excluded
  only from the supervised classifier dataset.
- Split by product_id with GroupShuffleSplit so comments from the same
  product cannot appear in both train and test.
- Match the existing TF-IDF baseline:
    test_size=0.2
    random_state=42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


VALID_LABELS = (
    "recommended",
    "not_recommended",
    "no_idea",
)

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "id",
    "product_id",
    "title",
    "body",
    "advantages",
    "disadvantages",
    "recommendation_status",
    "text",
}


def _validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Processed comments dataset is missing required columns: "
            + ", ".join(missing)
        )


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = (
        df["recommendation_status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    result: dict[str, int] = {}
    for label, count in counts.items():
        key = "__missing__" if pd.isna(label) else str(label)
        result[key] = int(count)

    return result


def _label_distribution(df: pd.DataFrame) -> dict[str, float]:
    dist = (
        df["recommendation_status"]
        .value_counts(normalize=True)
        .reindex(VALID_LABELS, fill_value=0.0)
    )
    return {str(k): float(v) for k, v in dist.items()}


def build_group_splits(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Create product-grouped train/test splits for recommendation prediction.

    Important:
    Missing recommendation_status rows are filtered only here for supervised
    modeling. They remain available in comments_processed.parquet for RAG,
    retrieval, and analytics.
    """
    _validate_schema(df)

    input_rows = len(df)
    input_status_counts = _label_counts(df)

    # --------------------------------------------------------------
    # 1) Keep ONLY the three official project labels.
    #    This correctly excludes NaN without confusing it with no_idea.
    # --------------------------------------------------------------
    labeled = df[
        df["recommendation_status"].isin(VALID_LABELS)
    ].copy()

    if labeled.empty:
        raise ValueError("No labeled rows found after filtering.")

    if labeled["product_id"].isna().any():
        raise ValueError(
            "product_id contains missing values in labeled rows. "
            "Cannot create leakage-free group split."
        )

    # Make target explicit/categorical without remapping its meaning.
    labeled["recommendation_status"] = pd.Categorical(
        labeled["recommendation_status"],
        categories=list(VALID_LABELS),
    )

    # --------------------------------------------------------------
    # 2) Exact split logic used by the existing TF-IDF baseline.
    # --------------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(
            labeled,
            groups=labeled["product_id"],
        )
    )

    train_df = labeled.iloc[train_idx].copy()
    test_df = labeled.iloc[test_idx].copy()

    # --------------------------------------------------------------
    # 3) Leakage assertions
    # --------------------------------------------------------------
    train_products = set(train_df["product_id"].unique())
    test_products = set(test_df["product_id"].unique())

    overlap = train_products.intersection(test_products)

    if overlap:
        preview = list(overlap)[:10]
        raise AssertionError(
            "DATA LEAKAGE: product_id appears in both train and test. "
            f"Example overlapping product IDs: {preview}"
        )

    # Every output row must still have one of the official classes.
    if not train_df["recommendation_status"].isin(VALID_LABELS).all():
        raise AssertionError("Invalid label found in train split.")

    if not test_df["recommendation_status"].isin(VALID_LABELS).all():
        raise AssertionError("Invalid label found in test split.")

    # All 3 classes should normally appear in each large split.
    train_classes = set(
        train_df["recommendation_status"].dropna().astype(str).unique()
    )
    test_classes = set(
        test_df["recommendation_status"].dropna().astype(str).unique()
    )

    missing_train_classes = sorted(set(VALID_LABELS) - train_classes)
    missing_test_classes = sorted(set(VALID_LABELS) - test_classes)

    if missing_train_classes:
        raise AssertionError(
            f"Train split is missing classes: {missing_train_classes}"
        )

    if missing_test_classes:
        raise AssertionError(
            f"Test split is missing classes: {missing_test_classes}"
        )

    report: dict[str, Any] = {
        "input_rows_processed_dataset": int(input_rows),
        "input_recommendation_status_counts": input_status_counts,
        "official_labels": list(VALID_LABELS),
        "labeled_rows_used_for_modeling": int(len(labeled)),
        "rows_excluded_missing_or_invalid_label": int(input_rows - len(labeled)),
        "split_method": "GroupShuffleSplit",
        "group_column": "product_id",
        "test_size": float(test_size),
        "random_state": int(random_state),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_unique_products": int(train_df["product_id"].nunique()),
        "test_unique_products": int(test_df["product_id"].nunique()),
        "product_overlap_count": int(len(overlap)),
        "train_label_counts": _label_counts(train_df),
        "test_label_counts": _label_counts(test_df),
        "train_label_distribution": _label_distribution(train_df),
        "test_label_distribution": _label_distribution(test_df),
    }

    return train_df, test_df, report


def save_splits(
    input_path: str | Path,
    train_output: str | Path,
    test_output: str | Path,
    report_output: str | Path | None = None,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    input_path = Path(input_path)
    train_output = Path(train_output)
    test_output = Path(test_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Processed dataset shape: {df.shape}")

    print("\nRecommendation status BEFORE classifier filtering:")
    print(df["recommendation_status"].value_counts(dropna=False))

    train_df, test_df, report = build_group_splits(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(
        train_output,
        index=False,
        compression="zstd",
    )
    test_df.to_parquet(
        test_output,
        index=False,
        compression="zstd",
    )

    if report_output is None:
        report_output = train_output.parent / "recommendation_split_report.json"
    else:
        report_output = Path(report_output)

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("RECOMMENDATION CLASSIFIER SPLIT")
    print("=" * 70)

    print(f"Labeled rows used : {len(train_df) + len(test_df):,}")
    print(f"Train rows        : {len(train_df):,}")
    print(f"Test rows         : {len(test_df):,}")
    print(f"Train products    : {train_df['product_id'].nunique():,}")
    print(f"Test products     : {test_df['product_id'].nunique():,}")
    print("Product overlap   : 0  ✅")

    print("\nTrain class counts:")
    print(train_df["recommendation_status"].value_counts())

    print("\nTest class counts:")
    print(test_df["recommendation_status"].value_counts())

    print("\nTrain class distribution:")
    print(
        train_df["recommendation_status"]
        .value_counts(normalize=True)
        .round(4)
    )

    print("\nTest class distribution:")
    print(
        test_df["recommendation_status"]
        .value_counts(normalize=True)
        .round(4)
    )

    print(f"\nSaved train split : {train_output}")
    print(f"Saved test split  : {test_output}")
    print(f"Saved report      : {report_output}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build product-grouped recommendation-status train/test splits."
        )
    )

    parser.add_argument(
        "--input",
        default="data/processed/comments_processed.parquet",
        help="Processed comments parquet.",
    )
    parser.add_argument(
        "--train-output",
        default="data/processed/train_group_split.parquet",
        help="Output train parquet.",
    )
    parser.add_argument(
        "--test-output",
        default="data/processed/test_group_split.parquet",
        help="Output test parquet.",
    )
    parser.add_argument(
        "--report-output",
        default="data/processed/recommendation_split_report.json",
        help="Output JSON report.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of product groups assigned to test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed.",
    )

    args = parser.parse_args()

    save_splits(
        input_path=args.input,
        train_output=args.train_output,
        test_output=args.test_output,
        report_output=args.report_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
