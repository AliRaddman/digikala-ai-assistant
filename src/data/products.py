"""Products cleaning pipeline: products_raw.parquet -> products_clean.parquet.

Owner: Ali. Reproducible from the CLI:

    python -m src.data.products --raw data/raw/products_raw.parquet \
        --out data/processed/products_clean_v1.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.normalize import build_search_text, normalize

RENAME: dict[str, str] = {
    "id": "product_id",
    "title_fa": "title",
    "Brand": "brand",
    "Price": "price",
    "Rate": "rate",
    "Rate_cnt": "rate_count",
    "Category1": "cat1",
    "Category2": "cat2",
    "sub_category": "sub_cat",
    "Seller": "seller",
    "Is_Fake": "is_fake",
}

OUTPUT_COLUMNS: list[str] = [
    "product_id",
    "title",
    "brand",
    "price",
    "rate",
    "rate_count",
    "cat1",
    "cat2",
    "sub_cat",
    "seller",
    "is_fake",
    "search_text",
]

_TEXT_COLUMNS = ("title", "brand", "cat1", "cat2", "sub_cat", "seller")
_BOOL_MAP = {
    True: True,
    False: False,
    1: True,
    0: False,
    "True": True,
    "False": False,
    "true": True,
    "false": False,
    "1": True,
    "0": False,
}


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw parquet and rename columns to the agreed schema."""
    df = pd.read_parquet(path)
    missing = set(RENAME) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")
    return df[list(RENAME)].rename(columns=RENAME)


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and turn placeholder values into nulls.

    Two substantive decisions live here. price <= 0 becomes null, because a
    zero price means "not on sale" rather than "free". rate becomes null
    wherever rate_count is 0, because Rate is stored out of 100 and an unrated
    product would otherwise look like the worst product in the catalogue to
    every downstream ranker and prompt.
    """
    df = df.copy()
    df["product_id"] = df["product_id"].astype("string").str.strip()

    price = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = price.where(price > 0)

    df["rate_count"] = (
        pd.to_numeric(df["rate_count"], errors="coerce").fillna(0).astype("int64")
    )
    rate = pd.to_numeric(df["rate"], errors="coerce")
    df["rate"] = rate.where((df["rate_count"] > 0) & rate.between(0, 100))

    df["is_fake"] = df["is_fake"].map(_BOOL_MAP).fillna(False).astype(bool)
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop exact duplicates, then keep one row per product_id.

    Ids that survive with conflicting content differ mostly in price, which
    looks like snapshots taken at different times. The lowest non-null price
    wins, since it sits closer to min_price_last_month. Ordering matters:
    cast_types must run first so a placeholder 0 price cannot win.
    """
    before = len(df)
    df = df.drop_duplicates()
    after_exact = len(df)

    df = (
        df.sort_values("price", na_position="last", kind="mergesort")
        .drop_duplicates(subset="product_id", keep="first")
        .sort_index()
    )
    stats = {
        "rows_in": before,
        "exact_duplicates_dropped": before - after_exact,
        "conflicting_id_rows_dropped": after_exact - len(df),
        "rows_out": len(df),
    }
    return df, stats


def add_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the display text columns and build search_text."""
    df = df.copy()
    for col in _TEXT_COLUMNS:
        df[col] = df[col].map(normalize).replace("", None)

    df["search_text"] = [
        build_search_text(title, brand, cat1, cat2, sub_cat, dedupe_tokens=False)
        for title, brand, cat1, cat2, sub_cat in zip(
            df["title"], df["brand"], df["cat1"], df["cat2"], df["sub_cat"]
        )
    ]
    return df


def build_products(
    raw_path: Path, out_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full pipeline and write the cleaned parquet."""
    df = load_raw(raw_path)
    df = cast_types(df)
    df, stats = deduplicate(df)
    df = add_text_fields(df)

    empty_title = df["search_text"].str.len() == 0
    stats["empty_title_dropped"] = int(empty_title.sum())
    df = df.loc[~empty_title, OUTPUT_COLUMNS].reset_index(drop=True)

    if df["product_id"].duplicated().any():
        raise AssertionError("product_id is not unique after deduplication")

    stats.update(
        {
            "final_rows": len(df),
            "null_price_pct": round(100 * df["price"].isna().mean(), 2),
            "null_rate_pct": round(100 * df["rate"].isna().mean(), 2),
            "null_brand_pct": round(100 * df["brand"].isna().mean(), 2),
            "null_cat2_pct": round(100 * df["cat2"].isna().mean(), 2),
            "is_fake_pct": round(100 * df["is_fake"].mean(), 2),
            "mean_search_tokens": round(
                df["search_text"].str.split().str.len().mean(), 2
            ),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the Digikala products table.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, stats = build_products(args.raw, args.out)
    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"{key:<{width}}  {value}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()