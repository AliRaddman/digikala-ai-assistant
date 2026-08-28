"""Build the per-product comment retrieval index.

Owner: Ali. Produces the artefacts CommentRetriever needs:

    comments_meta_v1.parquet             row order + fields for Evidence/filters
    comments_emb_e5base_v1.npy           embeddings, same row order as meta
    comments_emb_e5base_v1.progress.json how far the encode got (resume marker)
    comments_product_map_v1.json         product_id -> [row indices into meta]
    comments_bm25_v1.npz                 pre-weighted sparse BM25 matrix
    comments_bm25_vocab_v1.json          term -> column

    python -m scripts.build_comment_index \
        --clean data/processed/comments_clean_v1.parquet \
        --out-dir data/indexes --max-per-product 50 --fp16

Design decision (see docs/DECISIONS.md, "لایه نظرات"): comment retrieval is
always scoped to one product ("ایراد این محصول چیست؟"), never a free sweep of
6M comments. A global ANN index (FAISS) is therefore the wrong tool here --
it cannot express "restrict to this product_id" as a hard constraint and
would force an over-fetch-then-filter dance for what is, per product, a
handful of vectors. Instead: embeddings live in a flat .npy read with
mmap_mode="r", a product_id -> row-indices map narrows a query to just that
product's rows, and scoring is one exact dot product over a few dozen
vectors -- exact, not approximate, and fast because the candidate set per
query is tiny.

Both the dense and BM25 indexes are built over the SAME capped subset (see
select_comments below), so they share one meta parquet and one row order.
comments_clean_v1.parquet itself is never capped -- section 4 (category
analytics) reads the full file directly.

Memory: a first attempt at the full encode died with the machine out of RAM.
Two causes, both fixed here. (1) The 5.4M-row cleaned frame and the 3.4M-row
selection stayed alive through the encode; they are now dropped as soon as
meta/BM25 are on disk, and the encoder texts are re-read from the meta
parquet (only title+body), which also guarantees the embedding row order
matches meta by construction. (2) SentenceTransformer.encode() accumulated
all 3.4M vectors in RAM (~10.5 GB) and np.save then made a second copy;
encoding now runs chunk by chunk straight into a float32 memmap on disk, so
peak RAM is one chunk, not the whole matrix.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

MODEL_ID = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "
MAX_SEQ_LEN = 128
BATCH_SIZE = 128
CHUNK_SIZE = 100_000

BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_MAX_PER_PRODUCT = 50

META_FILENAME = "comments_meta_v1.parquet"
MAP_FILENAME = "comments_product_map_v1.json"
BM25_FILENAME = "comments_bm25_v1.npz"
VOCAB_FILENAME = "comments_bm25_vocab_v1.json"
EMB_FILENAME = "comments_emb_e5base_v1.npy"
PROGRESS_FILENAME = "comments_emb_e5base_v1.progress.json"

META_COLUMNS = [
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
]


def select_comments(df: pd.DataFrame, max_per_product: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Cap each product to its top `max_per_product` comments.

    Priority: more `likes` first (a community-validated usefulness signal,
    and sparse enough that a real difference should win outright), then
    longer `body` as the tiebreaker -- which is what decides the order for
    the very common case of several comments tied at 0 likes.
    """
    body_length = df["body"].str.len()
    ordered = (
        df.assign(_body_length=body_length)
        .sort_values(["likes", "_body_length"], ascending=[False, False], kind="mergesort")
    )
    group_sizes = ordered.groupby("product_id", sort=False).size()
    capped_products = group_sizes[group_sizes > max_per_product]

    selected = (
        ordered.groupby("product_id", sort=False)
        .head(max_per_product)
        .drop(columns="_body_length")
        .reset_index(drop=True)
    )

    excluded_rows = len(df) - len(selected)
    stats = {
        "max_per_product": max_per_product,
        "products_total": int(group_sizes.size),
        "products_capped": int(capped_products.size),
        "comments_in_capped_products": int(capped_products.sum()),
        "comments_excluded_by_cap": int(excluded_rows),
        "comments_excluded_pct": round(100 * excluded_rows / len(df), 2) if len(df) else 0.0,
        "rows_selected": len(selected),
    }
    return selected, stats


def build_dense_text(df: pd.DataFrame) -> list[str]:
    """Natural-language text for the encoder: title + body.

    Unlike search_text (lowercased, punctuation-stripped, built for BM25),
    title/body here are already the display-normalized columns from
    comments.py -- case and punctuation intact -- which is what a
    contrastively-trained sentence encoder was benchmarked on.
    """
    texts = []
    for title, body in zip(df["title"], df["body"]):
        parts = [title] if isinstance(title, str) and title else []
        parts.append(body)
        texts.append(" | ".join(parts))
    return texts


def build_bm25(docs: list[str]) -> tuple[sparse.csc_matrix, dict[str, int]]:
    """Same pre-weighted BM25 construction as scripts/build_index.py."""
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

    avgdl = float(lengths.mean()) if n_docs else 1.0
    norm = (BM25_K1 * (1 - BM25_B + BM25_B * lengths / avgdl)).astype(np.float32)

    tf = tf.tocoo()
    weights = (
        idf[tf.col] * tf.data * (BM25_K1 + 1) / (tf.data + norm[tf.row])
    ).astype(np.float32)
    return (
        sparse.csc_matrix((weights, (tf.row, tf.col)), shape=(n_docs, n_terms)),
        vocab,
    )


def build_product_map(product_ids: pd.Series) -> dict[str, list[int]]:
    """product_id -> row indices into the meta/embedding row order."""
    mapping: dict[str, list[int]] = {}
    for position, product_id in enumerate(product_ids):
        mapping.setdefault(product_id, []).append(position)
    return mapping


def build_index_tables(
    clean_path: Path, out_dir: Path, max_per_product: int, *, skip_sparse: bool
) -> dict[str, object]:
    """Write meta, product map and BM25, then release everything large.

    Nothing built here is returned as a DataFrame on purpose: the encode step
    re-reads what it needs from the meta parquet, so the multi-gigabyte
    frames die here rather than living alongside the model.
    """
    df = pd.read_parquet(clean_path)
    df["comment_id"] = df["comment_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    print(f"{len(df)} comments, {df['product_id'].nunique()} products", flush=True)

    selected, stats = select_comments(df, max_per_product)
    del df
    gc.collect()
    for key, value in stats.items():
        print(f"{key:<28} {value}", flush=True)

    meta_path = out_dir / META_FILENAME
    selected[META_COLUMNS].to_parquet(meta_path, compression="zstd", index=False)
    print(f"meta -> {meta_path}", flush=True)

    map_path = out_dir / MAP_FILENAME
    map_path.write_text(
        json.dumps(build_product_map(selected["product_id"]), ensure_ascii=False)
    )
    print(f"product map -> {map_path}", flush=True)

    if not skip_sparse:
        start = time.perf_counter()
        # search_text was already built by comments.py; reuse it as-is.
        matrix, vocab = build_bm25(selected["search_text"].fillna("").tolist())
        sparse_path = out_dir / BM25_FILENAME
        sparse.save_npz(sparse_path, matrix)
        (out_dir / VOCAB_FILENAME).write_text(json.dumps(vocab, ensure_ascii=False))
        stats["bm25_seconds"] = round(time.perf_counter() - start, 1)
        stats["vocab_size"] = len(vocab)
        stats["bm25_mb"] = round(sparse_path.stat().st_size / 1e6, 1)
        print(f"bm25 -> {sparse_path}", flush=True)
        del matrix, vocab
        gc.collect()

    del selected
    gc.collect()
    return stats


def load_dense_texts(meta_path: Path) -> list[str]:
    """Encoder inputs, read back from meta so the row order cannot drift."""
    df = pd.read_parquet(meta_path, columns=["title", "body"])
    texts = build_dense_text(df)
    del df
    gc.collect()
    return texts


def load_encoder(model_id: str, *, fp16: bool):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    if fp16:
        if device == "cuda":
            model = model.half()
        else:
            print("--fp16 ignored: half precision needs CUDA", flush=True)
    return model


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _read_progress(progress_path: Path, total_rows: int, dim: int) -> int:
    """Rows already encoded, or 0 when there is nothing valid to resume from."""
    if not progress_path.exists():
        return 0
    try:
        progress = json.loads(progress_path.read_text())
    except json.JSONDecodeError:
        return 0
    if progress.get("total_rows") != total_rows or progress.get("dim") != dim:
        print(
            f"progress file describes {progress.get('total_rows')} rows x "
            f"{progress.get('dim')} dims, expected {total_rows} x {dim} "
            "-- restarting the encode from scratch",
            flush=True,
        )
        return 0
    return int(progress.get("rows_written", 0))


def encode_to_memmap(
    texts: list[str],
    emb_path: Path,
    progress_path: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = CHUNK_SIZE,
    fp16: bool = False,
) -> dict[str, object]:
    """Encode `texts` chunk by chunk straight into a float32 memmap.

    The matrix is never held in RAM: each chunk is written through the memmap
    and dropped, and a sidecar progress file records how far the run got so a
    killed process resumes from that row instead of re-encoding everything.
    The stored dtype stays float32 even under --fp16 (which only speeds up
    the forward pass), so CommentRetriever's dot product is unaffected.
    """
    total_rows = len(texts)
    model = load_encoder(model_id, fp16=fp16)
    dim = int(model.get_sentence_embedding_dimension())

    start_row = _read_progress(progress_path, total_rows, dim)
    if start_row and not emb_path.exists():
        start_row = 0
    if start_row >= total_rows and emb_path.exists():
        print(f"embeddings already complete at {emb_path}", flush=True)
        return {"encode_seconds": 0.0, "dim": dim, "vectors": total_rows, "resumed_from": start_row}

    per_chunk_bytes = chunk_size * dim * (6 if fp16 else 4)
    print(
        f"rows={total_rows} dim={dim} batch_size={batch_size} chunk_size={chunk_size} "
        f"fp16={fp16}",
        flush=True,
    )
    print(
        f"estimated memory: {total_rows * dim * 4 / 1e9:.1f} GB on disk (memmap), "
        f"~{per_chunk_bytes / 1e9:.2f} GB RAM per chunk plus the model",
        flush=True,
    )
    if start_row:
        print(f"resuming from row {start_row} ({100 * start_row / total_rows:.1f}%)", flush=True)

    if start_row:
        embeddings = np.lib.format.open_memmap(emb_path, mode="r+")
        if embeddings.shape != (total_rows, dim):
            raise ValueError(
                f"existing {emb_path} has shape {embeddings.shape}, expected "
                f"{(total_rows, dim)}; delete it to rebuild from scratch"
            )
    else:
        embeddings = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype="float32", shape=(total_rows, dim)
        )

    started = time.perf_counter()
    rows_this_run = 0
    for start in range(start_row, total_rows, chunk_size):
        stop = min(start + chunk_size, total_rows)
        chunk = [PASSAGE_PREFIX + text for text in texts[start:stop]]
        vectors = model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings[start:stop] = vectors.astype("float32", copy=False)
        embeddings.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "rows_written": stop,
                    "total_rows": total_rows,
                    "dim": dim,
                    "model": model_id,
                    "fp16": fp16,
                },
                indent=2,
            )
        )
        del chunk, vectors
        gc.collect()

        rows_this_run += stop - start
        elapsed = time.perf_counter() - started
        rate = rows_this_run / elapsed if elapsed else 0.0
        remaining = total_rows - stop
        eta = _format_duration(remaining / rate) if rate else "unknown"
        print(
            f"  {stop}/{total_rows} ({100 * stop / total_rows:5.1f}%) "
            f"{rate:,.0f} rows/s  elapsed {_format_duration(elapsed)}  eta {eta}",
            flush=True,
        )

    encode_seconds = time.perf_counter() - started
    del embeddings
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    return {
        "encode_seconds": round(encode_seconds, 1),
        "dim": dim,
        "vectors": total_rows,
        "resumed_from": start_row,
        "emb_mb": round(emb_path.stat().st_size / 1e6, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the comment retrieval indexes.")
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--max-per-product", type=int, default=DEFAULT_MAX_PER_PRODUCT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Rows encoded per memmap write. Bounds peak RAM.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run the encoder in half precision (CUDA only). Output stays float32.",
    )
    parser.add_argument(
        "--rebuild-tables",
        action="store_true",
        help="Rebuild meta/product map/BM25 even when they already exist on disk.",
    )
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    args = parser.parse_args()

    if args.max_per_product < 1:
        raise ValueError("--max-per-product must be at least 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / META_FILENAME
    stats_path = args.out_dir / "comment_index_stats_v1.json"
    tables_exist = meta_path.exists() and (args.out_dir / MAP_FILENAME).exists()

    stats: dict[str, object] = {}
    if args.rebuild_tables or not tables_exist:
        stats.update(
            build_index_tables(
                args.clean,
                args.out_dir,
                args.max_per_product,
                skip_sparse=args.skip_sparse,
            )
        )
    else:
        print(
            f"reusing existing {META_FILENAME} / {MAP_FILENAME} / {BM25_FILENAME} "
            "(pass --rebuild-tables to rebuild)",
            flush=True,
        )
        # Carry the previous run's selection/BM25 numbers forward: they
        # describe the tables still on disk, and docs/DECISIONS.md cites them.
        if stats_path.exists():
            try:
                stats.update(json.loads(stats_path.read_text()))
            except json.JSONDecodeError:
                pass

    if not args.skip_dense:
        texts = load_dense_texts(meta_path)
        stats.update(
            encode_to_memmap(
                texts,
                args.out_dir / EMB_FILENAME,
                args.out_dir / PROGRESS_FILENAME,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                fp16=args.fp16,
            )
        )
        del texts
        gc.collect()
        print(f"embeddings -> {args.out_dir / EMB_FILENAME}", flush=True)

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    for key, value in stats.items():
        print(f"{key:<28} {value}", flush=True)


if __name__ == "__main__":
    main()
