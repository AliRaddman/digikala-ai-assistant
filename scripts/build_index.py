"""Build the product indexes over the full catalogue.

Owner: Ali. Produces three artefacts that together are everything the
retriever needs, and that fit on the shared Drive:

    products_meta_v1.parquet   row order + the fields used for filtering
    products_bm25_v1.npz       pre-weighted sparse BM25 matrix + vocabulary
    products_e5base_v1.faiss   dense index

    python -m scripts.build_index \
        --clean data/processed/products_clean_v1.parquet \
        --out-dir data/indexes --index-type ivfpq

Build `flat` once as the quality reference and `ivfpq` for everyday use: the
flat index is exact but ~2.9 GB, which cannot realistically be moved over the
Drive, while ivfpq lands near 100 MB. The recall lost to quantisation is
measured, not assumed — score both against the eval set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.normalize import tokenize

MODEL_ID = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "
MAX_SEQ_LEN = 128
BATCH_SIZE = 128

BM25_K1 = 1.5
BM25_B = 0.75

IVF_NLIST = 4096
PQ_M = 96
PQ_BITS = 8
TRAIN_SAMPLE = 200_000

META_COLUMNS = [
    "product_id",
    "title",
    "brand",
    "price",
    "rate",
    "rate_count",
    "cat1",
    "cat2",
    "sub_cat",
    "is_fake",
]


def build_dense_text(df: pd.DataFrame) -> list[str]:
    """Same construction as the benchmark, so the chosen model sees the text
    it was compared on."""
    texts = []
    for title, brand, cat2, cat1 in zip(df["title"], df["brand"], df["cat2"], df["cat1"]):
        parts = [title]
        if isinstance(brand, str) and brand and brand != "متفرقه":
            parts.append(brand)
        for value in (cat2, cat1):
            if isinstance(value, str) and value:
                parts.append(value)
                break
        texts.append(" | ".join(parts))
    return texts


def build_bm25(docs: list[str]) -> tuple[sparse.csc_matrix, dict[str, int]]:
    """Pre-weighted BM25 matrix.

    Each cell holds the full BM25 contribution of one term to one document:

        w(t,d) = idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·dl(d)/avgdl))
        idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

    Baking the weights in once means scoring a query is just summing a few
    columns, which keeps latency in the low milliseconds without a search
    server. The cost is that k1 and b are frozen at build time.
    """
    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    freqs: list[int] = []
    lengths = np.zeros(len(docs), dtype=np.float32)

    for doc_id, doc in enumerate(docs):
        counts: dict[int, int] = {}
        tokens = doc.split()
        lengths[doc_id] = len(tokens)
        for token in tokens:
            term_id = vocab.setdefault(token, len(vocab))
            counts[term_id] = counts.get(term_id, 0) + 1
        rows.extend([doc_id] * len(counts))
        cols.extend(counts.keys())
        freqs.extend(counts.values())

    n_docs, n_terms = len(docs), len(vocab)
    tf = sparse.csr_matrix(
        (np.array(freqs, dtype=np.float32), (rows, cols)), shape=(n_docs, n_terms)
    )

    df_counts = np.asarray((tf > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (n_docs - df_counts + 0.5) / (df_counts + 0.5)).astype(np.float32)

    avgdl = float(lengths.mean())
    norm = (BM25_K1 * (1 - BM25_B + BM25_B * lengths / avgdl)).astype(np.float32)

    tf = tf.tocoo()
    weights = (
        idf[tf.col] * tf.data * (BM25_K1 + 1) / (tf.data + norm[tf.row])
    ).astype(np.float32)
    matrix = sparse.csc_matrix(
        (weights, (tf.row, tf.col)), shape=(n_docs, n_terms)
    )
    return matrix, vocab


def build_dense(texts: list[str], index_type: str, out_path: Path) -> dict[str, float]:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.max_seq_length = MAX_SEQ_LEN

    start = time.perf_counter()
    embeddings = model.encode(
        [PASSAGE_PREFIX + text for text in texts],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    encode_seconds = time.perf_counter() - start

    dim = embeddings.shape[1]
    start = time.perf_counter()
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantiser, dim, IVF_NLIST, PQ_M, PQ_BITS, faiss.METRIC_INNER_PRODUCT
        )
        sample = embeddings
        if len(embeddings) > TRAIN_SAMPLE:
            rng = np.random.default_rng(42)
            sample = embeddings[rng.choice(len(embeddings), TRAIN_SAMPLE, replace=False)]
        index.train(sample)

    index.add(embeddings)
    build_seconds = time.perf_counter() - start
    faiss.write_index(index, str(out_path))

    del model
    torch.cuda.empty_cache()
    return {
        "encode_seconds": round(encode_seconds, 1),
        "index_seconds": round(build_seconds, 1),
        "dim": dim,
        "vectors": int(index.ntotal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the product indexes.")
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--index-type", choices=["flat", "ivfpq"], default="ivfpq")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.clean)
    df["product_id"] = df["product_id"].astype(str)
    print(f"{len(df)} products")

    meta_path = args.out_dir / "products_meta_v1.parquet"
    df[META_COLUMNS].to_parquet(meta_path, compression="zstd", index=False)
    print(f"meta -> {meta_path}")

    stats: dict[str, object] = {"rows": len(df)}

    if not args.skip_sparse:
        start = time.perf_counter()
        matrix, vocab = build_bm25(df["search_text"].tolist())
        sparse_path = args.out_dir / "products_bm25_v1.npz"
        sparse.save_npz(sparse_path, matrix)
        (args.out_dir / "products_bm25_vocab_v1.json").write_text(
            json.dumps(vocab, ensure_ascii=False)
        )
        stats["bm25_seconds"] = round(time.perf_counter() - start, 1)
        stats["vocab_size"] = len(vocab)
        stats["bm25_mb"] = round(sparse_path.stat().st_size / 1e6, 1)
        print(f"bm25 -> {sparse_path}")

    if not args.skip_dense:
        dense_path = args.out_dir / f"products_e5base_{args.index_type}_v1.faiss"
        stats.update(build_dense(build_dense_text(df), args.index_type, dense_path))
        stats["dense_mb"] = round(dense_path.stat().st_size / 1e6, 1)
        print(f"dense -> {dense_path}")

    (args.out_dir / "index_stats_v1.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    for key, value in stats.items():
        print(f"{key:<16} {value}")


if __name__ == "__main__":
    main()