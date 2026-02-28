#!/usr/bin/env python3
"""
v3 knowledge index builder

- Consumes page-level corpus from OCR pipeline
- Filters low-quality/noisy pages
- Rebuilds deduplicated chunks
- Builds dense embeddings for retrieval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u00ad", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(
        r"(?:[가-힣]\s+){2,}[가-힣]",
        lambda m: re.sub(r"\s+", "", m.group(0)),
        t,
    )
    return t


def split_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    n = len(text)
    if n <= chunk_size:
        return [text]
    out: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if end < n:
            # prefer sentence/space boundary near the end
            last_break = max(chunk.rfind(". "), chunk.rfind(" "), chunk.rfind("\n"))
            if last_break > int(chunk_size * 0.6):
                end = start + last_break
                chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap)
    return out


def pick_major_sources(rows: List[Dict[str, Any]], min_pages: int) -> Set[str]:
    counts = Counter(str(r.get("source_file", "")) for r in rows)
    major = {k for k, v in counts.items() if v >= min_pages}
    if not major:
        major = set(counts.keys())
    return major


def build_clean_chunks(
    corpus_rows: List[Dict[str, Any]],
    min_pages_per_source: int,
    min_quality: float,
    min_chars: int,
    chunk_size: int,
    overlap: int,
    keep_methods: List[str],
) -> List[Dict[str, Any]]:
    major_sources = pick_major_sources(corpus_rows, min_pages=min_pages_per_source)
    keep_methods_set = {m.strip().lower() for m in keep_methods if m.strip()}

    seen_hash: Set[str] = set()
    chunks: List[Dict[str, Any]] = []

    for r in corpus_rows:
        source = str(r.get("source_file", ""))
        if source not in major_sources:
            continue

        method = str(r.get("method", "")).lower()
        if keep_methods_set and method and not any(method.startswith(x) for x in keep_methods_set):
            continue

        q = float(r.get("quality_score", 0.0) or 0.0)
        chars = int(r.get("char_count", 0) or 0)
        if q < min_quality or chars < min_chars:
            continue

        page = int(r.get("page", 0) or 0)
        txt = normalize_text(str(r.get("text", "")))
        if len(txt) < min_chars:
            continue

        pieces = split_chunks(txt, chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces, start=1):
            h = hashlib.sha1(piece.encode("utf-8", errors="ignore")).hexdigest()
            if h in seen_hash:
                continue
            seen_hash.add(h)

            key = f"{source}::{page}"
            key_hash = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
            chunks.append(
                {
                    "chunk_id": f"{key_hash}_p{page}_c{idx}",
                    "source_file": source,
                    "page_start": page,
                    "page_end": page,
                    "method": method,
                    "text": piece,
                }
            )
    return chunks


def embed_chunks(
    model_name: str,
    texts: List[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        [f"passage: {t}" for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32, copy=False)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-pages-per-source", type=int, default=120)
    ap.add_argument("--min-quality", type=float, default=0.82)
    ap.add_argument("--min-chars", type=int, default=160)
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=140)
    ap.add_argument(
        "--keep-methods",
        default="textlayer,ocr",
        help="comma list prefix filter; e.g. textlayer,ocr",
    )
    args = ap.parse_args()

    corpus_path = Path(args.corpus_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"missing corpus file: {corpus_path}")

    rows = read_jsonl(corpus_path)
    if not rows:
        raise RuntimeError("corpus is empty")

    keep_methods = [x.strip() for x in str(args.keep_methods).split(",") if x.strip()]

    chunks = build_clean_chunks(
        corpus_rows=rows,
        min_pages_per_source=max(1, int(args.min_pages_per_source)),
        min_quality=float(args.min_quality),
        min_chars=max(1, int(args.min_chars)),
        chunk_size=max(320, int(args.chunk_size)),
        overlap=max(32, int(args.chunk_overlap)),
        keep_methods=keep_methods,
    )
    if not chunks:
        raise RuntimeError("no chunks after filtering; lower thresholds")

    chunks_path = out_dir / "chunks_clean.jsonl"
    write_jsonl(chunks_path, chunks)

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = [c["text"] for c in chunks]
    emb = embed_chunks(
        model_name=args.embed_model,
        texts=texts,
        batch_size=max(1, int(args.batch_size)),
        device=device,
    )

    emb_path = out_dir / "embeddings.npy"
    np.save(str(emb_path), emb.astype(np.float16))

    meta = {
        "status": "ok",
        "chunks": len(chunks),
        "embed_dim": int(emb.shape[1]),
        "embed_model": args.embed_model,
        "device": device,
        "corpus_path": str(corpus_path),
        "chunks_path": str(chunks_path),
        "embeddings_path": str(emb_path),
        "filters": {
            "min_pages_per_source": args.min_pages_per_source,
            "min_quality": args.min_quality,
            "min_chars": args.min_chars,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "keep_methods": keep_methods,
        },
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
