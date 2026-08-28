"""
Comments cleaning pipeline for Digikala AI Assistant.

Important project decisions
---------------------------
1. recommendation_status is NEVER converted to sentiment and NEVER filled.
   The three real labels remain:
       recommended / not_recommended / no_idea
   Missing recommendation_status remains missing (NaN / <NA>).

2. Persian text normalization is delegated to the repository's shared
   normalizer:
       from src.data.normalize import normalize
   No second/custom normalization implementation is defined here.

3. title, body, advantages and disadvantages remain separate columns.
   A combined `text` / `review_text` representation is added only as an
   extra field for classifiers/retrieval; it does not replace the originals.

4. Rows are not removed merely because recommendation_status is missing.
   Filtering to labeled rows belongs to the downstream classifier dataset.

5. Very short reviews are reported, not blindly deleted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.normalize import normalize


VALID_RECOMMENDATION_STATUSES = {
    "recommended",
    "not_recommended",
    "no_idea",
}

TEXT_COLUMNS = [
    "title",
    "body",
    "advantages",
    "disadvantages",
]

REQUIRED_COLUMNS = [
    "id",
    "title",
    "body",
    "created_at",
    "rate",
    "recommendation_status",
    "is_buyer",
    "product_id",
    "advantages",
    "disadvantages",
    "likes",
    "dislikes",
    "seller_title",
    "seller_code",
    "true_to_size_rate",
]


def _validate_input_columns(df: pd.DataFrame) -> None:
    """Fail early if the raw dataset schema is not what the project expects."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in comments dataset: "
            + ", ".join(missing)
        )


def _status_counts(series: pd.Series) -> dict[str, int]:
    """
    Return recommendation_status counts while explicitly keeping missing
    separate from the real `no_idea` class.
    """
    counts: dict[str, int] = {}

    for value, count in series.value_counts(dropna=False).items():
        key = "__missing__" if pd.isna(value) else str(value)
        counts[key] = int(count)

    return counts


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all searchable Persian text with the exact shared repository
    normalizer.

    We intentionally do not replace ZWNJ / half-space ourselves here.
    Search/indexing and cleaning must use the same normalize() behavior.
    """
    for col in TEXT_COLUMNS:
        # normalize() expects text. Empty source fields become empty strings.
        df[col] = df[col].fillna("").astype(str)

        # This is intentionally the project's shared normalizer:
        df[col] = df[col].map(normalize)

        # Defensive cleanup only for null return values; no custom text
        # normalization or ZWNJ manipulation is performed.
        df[col] = df[col].fillna("")

    return df


def _build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a combined normalized representation for TF-IDF / retrieval while
    preserving title/body/advantages/disadvantages separately.
    """
    combined = (
        df["title"]
        .str.cat(df["body"], sep=" ")
        .str.cat(df["advantages"], sep=" ")
        .str.cat(df["disadvantages"], sep=" ")
        .str.strip()
    )

    # `text` keeps compatibility with the current TF-IDF notebook.
    df["text"] = combined

    # `review_text` is a clearer alias for downstream RAG code.
    df["review_text"] = combined

    return df


def clean_comments_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean the raw comments dataframe without destroying downstream signals.

    Returns
    -------
    cleaned_df
        Full processed comments dataset. Missing recommendation labels remain.
    report
        Counts useful for validating that three classes and missing values
        stayed distinct.
    """
    _validate_input_columns(df)

    report: dict[str, Any] = {
        "input_rows": int(len(df)),
        "recommendation_status_before": _status_counts(
            df["recommendation_status"]
        ),
    }

    # Work on a copy so callers do not get unexpected mutation.
    df = df.copy()

    # ------------------------------------------------------------------
    # 1) Remove duplicate comment IDs
    # ------------------------------------------------------------------
    duplicate_id_count = int(df.duplicated(subset=["id"]).sum())
    df = df.drop_duplicates(subset=["id"], keep="first").copy()

    report["duplicate_comment_ids_removed"] = duplicate_id_count

    # ------------------------------------------------------------------
    # 2) Normalize text using ONLY src.data.normalize.normalize
    # ------------------------------------------------------------------
    df = _normalize_text_columns(df)

    # ------------------------------------------------------------------
    # 3) Remove only records with absolutely no usable text anywhere.
    #    Do NOT remove a row just because body is empty if title/advantages/
    #    disadvantages still contain useful evidence.
    # ------------------------------------------------------------------
    has_any_text = df[TEXT_COLUMNS].ne("").any(axis=1)
    no_text_count = int((~has_any_text).sum())

    df = df.loc[has_any_text].copy()
    report["rows_removed_no_text_anywhere"] = no_text_count

    # Short reviews are useful signals in many cases ("عالیه", "نخرید").
    # We only report them; we do not delete them.
    report["body_shorter_than_10_chars_kept"] = int(
        ((df["body"].str.len() > 0) & (df["body"].str.len() < 10)).sum()
    )

    # ------------------------------------------------------------------
    # 4) Preserve advantages/disadvantages separately AND add combined text
    # ------------------------------------------------------------------
    df = _build_combined_text(df)

    # ------------------------------------------------------------------
    # 5) recommendation_status: PRESERVE, DO NOT FILL, DO NOT REMAP
    # ------------------------------------------------------------------
    # This line is intentionally NOT present:
    # df["recommendation_status"] = df["recommendation_status"].fillna("no_idea")
    #
    # And there is intentionally no positive/negative sentiment mapping.

    non_null_statuses = set(
        df["recommendation_status"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    unexpected = sorted(
        non_null_statuses - VALID_RECOMMENDATION_STATUSES
    )

    report["unexpected_recommendation_status_values"] = unexpected
    report["recommendation_status_after"] = _status_counts(
        df["recommendation_status"]
    )

    report["labeled_rows_for_classifier"] = int(
        df["recommendation_status"].isin(
            VALID_RECOMMENDATION_STATUSES
        ).sum()
    )
    report["missing_recommendation_status_rows_kept"] = int(
        df["recommendation_status"].isna().sum()
    )
    report["output_rows"] = int(len(df))

    # Keep original project columns first, then the two extra text columns.
    output_columns = REQUIRED_COLUMNS + ["text", "review_text"]
    df = df[output_columns]

    return df, report


def process_comments_file(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read raw parquet, clean it, and save the processed parquet + report."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Raw shape: {df.shape}")

    cleaned, report = clean_comments_dataframe(df)

    print(f"Processed shape: {cleaned.shape}")
    print("recommendation_status:")
    print(cleaned["recommendation_status"].value_counts(dropna=False))

    # Parquet is already used by the rest of the project.
    cleaned.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    if report_path is None:
        report_path = output_path.with_name(
            output_path.stem + "_cleaning_report.json"
        )
    else:
        report_path = Path(report_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved processed dataset: {output_path}")
    print(f"Saved cleaning report:  {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Digikala comments without losing project labels."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw/comments_raw.parquet",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to processed/comments_processed.parquet",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for cleaning report JSON",
    )

    args = parser.parse_args()

    process_comments_file(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
