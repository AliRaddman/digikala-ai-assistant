"""Comments cleaning pipeline: comments_raw.parquet -> comments_clean.parquet.

Owner: Ali (reassigned from the comments layer, which had no owner). Reproducible
from the CLI:

    python -m src.data.comments --raw data/raw/comments_raw.parquet \
        --out data/processed/comments_clean_v1.parquet
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.normalize import normalize, to_search_text

RENAME: dict[str, str] = {"id": "comment_id"}

_RAW_COLUMNS = [
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
]

OUTPUT_COLUMNS: list[str] = [
    "comment_id",
    "product_id",
    "title",
    "body",
    "advantages",
    "disadvantages",
    "rate",
    "recommendation_status",
    "is_buyer",
    "likes",
    "dislikes",
    "created_at",
    "search_text",
]

_TEXT_COLUMNS = ("title", "advantages", "disadvantages")
MIN_BODY_LEN = 10

# rate here is a 1-5 star rating the user attached to their comment, unlike
# the products table's rate which is 0-100. Same column name, different
# table, different scale -- do not reuse products.py's rate logic here.
#
# The valid floor is 1, not 0: a stored 0 means "no stars given", the same
# placeholder the products table uses (see docs/DECISIONS.md). 484,876
# comments (9.0%) carry it, and 80% of them are recommendation_status
# "recommended" -- a reviewer writing "کیفیتش خیلی خوبه، راضی‌ام" and
# recommending the product did not award it zero stars.
RATE_MIN, RATE_MAX = 1, 5

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
    """Read the raw parquet, rename `id` -> `comment_id`, cast the two join keys.

    Ids are cast to string immediately, before any dedup or join, so a mixed
    int/string column can never make `drop_duplicates(subset="comment_id")` or
    a downstream products join fail silently.
    """
    df = pd.read_parquet(path)
    missing = set(_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")
    df = df[_RAW_COLUMNS].rename(columns=RENAME)
    df["comment_id"] = df["comment_id"].astype("string").str.strip()
    df["product_id"] = df["product_id"].astype("string").str.strip()
    return df


def drop_missing_body(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Comments with no body carry no evidence and cannot be cited."""
    missing = df["body"].isna() | (df["body"].astype("string").str.strip() == "")
    return df.loc[~missing].copy(), int(missing.sum())


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with a repeated comment_id, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset="comment_id", keep="first")
    return df, before - len(df)


def _join_list_field(value: object) -> object:
    """advantages/disadvantages sometimes arrive as a list of short phrases,
    and in this raw export specifically as a *string* holding the printed
    repr of a Python list (e.g. "['قیمت بالا']") rather than a native list
    column. Both forms are joined into one plain, normalizable string.
    """
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            value = parsed
    if isinstance(value, (list, tuple, np.ndarray)):
        return "، ".join(str(v).strip() for v in value if v not in (None, "", "nan"))
    return value


def add_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text columns and build search_text from title + body only.

    advantages/disadvantages are free-form, short bullet points ("سبک بود",
    "جنسش بد بود") and stay out of search_text: mixed into the body they would
    inflate term frequency for whatever adjective a user happened to write,
    distorting BM25 without adding real retrieval signal. They are kept as
    separate output columns because section 4 (category analytics) needs them
    apart from body to count recurring complaints.
    """
    df = df.copy()
    for col in _TEXT_COLUMNS:
        df[col] = df[col].map(_join_list_field).map(normalize).replace("", None)
    df["body"] = df["body"].map(_join_list_field).map(normalize)

    df["search_text"] = [
        to_search_text(f"{title or ''} {body}")
        for title, body in zip(df["title"], df["body"])
    ]
    return df


def filter_short_body(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop comments whose normalized body is under MIN_BODY_LEN characters.

    One- and two-word comments ("عالی", "خوبه") carry no answerable content:
    a QA chain retrieving them just spends evidence budget on noise, and in
    dense retrieval a cluster of near-duplicate one-word comments crowds out
    comments that actually explain a pro or con.
    """
    short = df["body"].str.len() < MIN_BODY_LEN
    return df.loc[~short].copy(), int(short.sum())


def cast_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce rate to numeric and null out anything outside 1-5.

    Comment rate is a 1-5 star rating, distinct from the products table's
    0-100 Rate column. Two kinds of value become null here:

    0 -- "no stars given", not "zero stars". Keeping it would drag every
    mean rating down and, worse, hand the QA chain "امتیاز: 0 از ۵" as
    evidence next to a review that recommends the product. This mirrors the
    products table decision in docs/DECISIONS.md.

    Out-of-range values like the 2500 seen in EDA, which are almost certainly
    products-scale values that leaked in, or data entry errors; either way
    not a valid star rating, so null rather than a guess.
    """
    df = df.copy()
    rate = pd.to_numeric(df["rate"], errors="coerce")
    df["rate"] = rate.where(rate.between(RATE_MIN, RATE_MAX))
    return df


def cast_remaining_types(df: pd.DataFrame) -> pd.DataFrame:
    """is_buyer to bool, likes/dislikes to non-negative int counts.

    A missing like/dislike count means the comment has zero votes, not an
    unknown number of votes, so it becomes 0 rather than staying null.
    recommendation_status is intentionally left untouched here: it must keep
    exactly its three raw string values plus NaN, since section 3 of the
    project trains directly on this column.
    """
    df = df.copy()
    df["is_buyer"] = df["is_buyer"].map(_BOOL_MAP).fillna(False).astype(bool)
    for col in ("likes", "dislikes"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df["recommendation_status"] = df["recommendation_status"].astype("string")
    return df


def build_comments(raw_path: Path, out_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full pipeline and write the cleaned parquet.

    Step order matches the agreed cleaning plan: drop rows with no body,
    dedupe by id, normalize text, build search_text, drop short bodies, then
    cast rate. Rate casting runs last because it is independent of the text
    steps and keeping it last means the short-body count is not affected by
    rows that would separately have had their rate nulled.
    """
    df = load_raw(raw_path)
    stats: dict[str, Any] = {"rows_in": len(df)}

    df, missing_body_dropped = drop_missing_body(df)
    stats["missing_body_dropped"] = missing_body_dropped

    df, duplicate_id_dropped = deduplicate(df)
    stats["duplicate_id_dropped"] = duplicate_id_dropped

    df = add_text_fields(df)

    df, short_body_dropped = filter_short_body(df)
    stats["short_body_dropped"] = short_body_dropped

    df = cast_rate(df)
    df = cast_remaining_types(df)

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    stats["rows_out"] = len(df)
    stats["unique_products"] = int(df["product_id"].nunique())
    stats["mean_body_length"] = round(float(df["body"].str.len().mean()), 2)
    stats["null_rate_pct"] = round(100 * df["rate"].isna().mean(), 2)
    stats["recommendation_status_distribution"] = {
        (key if key is not pd.NA else "NaN"): int(value)
        for key, value in df["recommendation_status"]
        .value_counts(dropna=False)
        .items()
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the Digikala comments table.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _, stats = build_comments(args.raw, args.out)
    for key, value in stats.items():
        if key == "recommendation_status_distribution":
            print(f"{key}:")
            for label, count in value.items():
                print(f"  {label:<16} {count}")
            continue
        print(f"{key:<32} {value}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
