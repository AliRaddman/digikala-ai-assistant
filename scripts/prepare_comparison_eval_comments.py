from pathlib import Path

import pandas as pd


SOURCE = Path("data/processed/comments_clean_v1.parquet")
OUTPUT = Path("data/processed/comments_comparison_eval.parquet")

N_PRODUCTS = 30
MIN_COMMENTS = 15
MAX_COMMENTS_PER_PRODUCT = 50


if not SOURCE.exists():
    raise FileNotFoundError(
        f"Missing source file: {SOURCE.resolve()}"
    )


df = pd.read_parquet(SOURCE)

df["comment_id"] = df["comment_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)

print("Source rows:", len(df))
print("Source products:", df["product_id"].nunique())


# Choose products with enough review evidence.
counts = (
    df.groupby("product_id")
    .size()
    .sort_values(ascending=False)
)

eligible = counts[
    counts >= MIN_COMMENTS
]

selected_product_ids = (
    eligible
    .head(N_PRODUCTS)
    .index
    .tolist()
)


subset = df[
    df["product_id"].isin(selected_product_ids)
].copy()


# Same general prioritization used by Ali's index builder:
# more likes first, then longer body.
subset["_body_length"] = (
    subset["body"]
    .fillna("")
    .astype(str)
    .str.len()
)

subset = (
    subset
    .sort_values(
        ["product_id", "likes", "_body_length"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    .groupby(
        "product_id",
        sort=False,
    )
    .head(MAX_COMMENTS_PER_PRODUCT)
    .drop(columns="_body_length")
    .reset_index(drop=True)
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

subset.to_parquet(
    OUTPUT,
    index=False,
)


print()
print("Saved:", OUTPUT)
print("Products:", subset["product_id"].nunique())
print("Comments:", len(subset))

print("\nSelected products:")
print(
    subset.groupby("product_id")
    .size()
    .sort_values(ascending=False)
)