"""Stratified product sample for the retrieval benchmark.

Owner: Ali. Encoding 948k products with three candidate models is not worth
the GPU time when the goal is only to rank the models against each other, so
the comparison runs on a sample.

    python -m src.data.sampling --clean data/processed/products_clean_v1.parquet \
        --out data/processed/products_sample_50k_v1.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STRATIFY_BY = "sub_cat"
DEFAULT_N = 50_000
DEFAULT_FLOOR = 2_000
SEED = 42


def _allocate(counts: pd.Series, n: int, floor: int) -> dict[str, int]:
    """Split n rows across strata: proportional, but with a per-stratum floor.

    Pure proportional allocation would give the travel stratum under a
    thousand rows, which is too thin to tell three models apart on travel
    queries. The floor keeps every stratum measurable; the remainder is taken
    from the largest strata, which can afford it.
    """
    total = int(counts.sum())
    alloc = {
        group: int(min(count, max(floor, round(n * count / total))))
        for group, count in counts.items()
    }

    order = list(counts.sort_values(ascending=False).index)
    for _ in range(1000):
        diff = sum(alloc.values()) - n
        if diff == 0:
            break
        for group in order:
            if diff > 0 and alloc[group] > floor:
                step = min(diff, alloc[group] - floor)
                alloc[group] -= step
                diff -= step
            elif diff < 0 and alloc[group] < counts[group]:
                step = min(-diff, int(counts[group]) - alloc[group])
                alloc[group] += step
                diff += step
            if diff == 0:
                break
    return alloc


def stratified_sample(
    df: pd.DataFrame,
    n: int = DEFAULT_N,
    by: str = STRATIFY_BY,
    floor: int = DEFAULT_FLOOR,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the sample and a per-stratum report of the allocation.

    Unrated products stay in. Dropping them would remove 62% of the catalogue
    and, because the missing ratings concentrate in clothing and books, the
    remaining sample would no longer resemble the index the system actually
    serves.
    """
    strata = df[by].fillna("unknown")
    counts = strata.value_counts()
    alloc = _allocate(counts, n=n, floor=floor)

    parts = [
        df.loc[strata == group].sample(size, random_state=seed)
        for group, size in alloc.items()
        if size > 0
    ]
    sample = pd.concat(parts).sort_index().reset_index(drop=True)

    report = pd.DataFrame(
        {
            "population": counts,
            "population_pct": (100 * counts / counts.sum()).round(2),
            "sampled": pd.Series(alloc),
        }
    )
    report["sampled_pct"] = (100 * report["sampled"] / report["sampled"].sum()).round(2)
    return sample, report.sort_values("population", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified sample of clean products.")
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--floor", type=int, default=DEFAULT_FLOOR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = pd.read_parquet(args.clean)
    sample, report = stratified_sample(
        df, n=args.n, by=STRATIFY_BY, floor=args.floor, seed=args.seed
    )

    print(report.to_string())
    print(f"\nrows          {len(sample)}")
    print(f"rated pct     {round(100 * sample['rate'].notna().mean(), 2)}")
    print(f"unique ids    {sample['product_id'].nunique()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, compression="zstd", index=False)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()