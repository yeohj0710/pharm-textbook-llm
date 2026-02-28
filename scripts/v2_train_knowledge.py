#!/usr/bin/env python3
"""
v2 training pipeline (single-turn QA architecture)

This does not fine-tune the generator directly.
Instead, it builds a high-quality grounded knowledge index:
1) filter/clean corpus pages
2) rebuild clean chunks
3) build dense embedding matrix for retrieval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u00ad", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()
    # spaced Hangul fix: "약 물 치 료" -> "약물치료"
    t = re.sub(
        r"(?:[가-힣]\s+){2,}[가-힣]",
        lambda m: re.sub(r"\s+", "", m.group(0)),
        t,
    )
    return t


def split_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    text = normalize_text(text)
    n = len(text)
    if n <= chunk_size:
        return [text]
    out: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def pick_major_sources(rows: List[Dict[str, Any]], min_pages: int) -> set[str]:
    c = Counter(str(r.get("source_file", "")) for r in rows)
    return {k for k, v in c.items() if v >= min_pages}


def build_clean_chunks(
    corpus_rows: List[Dict[str, Any]],
    min_pages_per_source: int,
    min_quality: float,
    min_chars: int,
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    major_sources = pick_major_sources(corpus_rows, min_pages=min_pages_per_source)
    if not major_sources:
        major_sources = set(str(r.get("source_file", "")) for r in corpus_rows)

    seen_hash: set[str] = set()
    chunks: List[Dict[str, Any]] = []

    for r in corpus_rows:
        source = str(r.get("source_file", ""))
        if source not in major_sources:
            continue
        q = float(r.get("quality_score", 0.0) or 0.0)
        ch = int(r.get("char_count", 0) or 0)
        if q < min_quality or ch < min_chars:
            continue
        page = int(r.get("page", 0) or 0)
        txt = normalize_text(str(r.get("text", "")))
        if len(txt) < min_chars:
            continue

        for idx, ck in enumerate(split_chunks(txt, chunk_size=chunk_size, overlap=overlap), start=1):
            h = hashlib.sha1(ck.encode("utf-8", errors="ignore")).hexdigest()
            if h in seen_hash:
                continue
            seen_hash.add(h)
            chunks.append(
                {
                    "chunk_id": f"{hashlib.sha1((source+str(page)).encode('utf-8',errors='ignore')).hexdigest()[:12]}_p{page}_c{idx}",
                    "source_file": source,
                    "page_start": page,
                    "page_end": page,
                    "text": ck,
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
    emb = emb.astype(np.float32, copy=False)
    return emb


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
    ap.add_argument("--min-pages-per-source", type=int, default=200)
    ap.add_argument("--min-quality", type=float, default=0.88)
    ap.add_argument("--min-chars", type=int, default=180)
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=140)
    args = ap.parse_args()

    corpus_path = Path(args.corpus_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"missing corpus file: {corpus_path}")

    rows = read_jsonl(corpus_path)
    if not rows:
        raise RuntimeError("corpus is empty")

    chunks = build_clean_chunks(
        corpus_rows=rows,
        min_pages_per_source=max(1, args.min_pages_per_source),
        min_quality=float(args.min_quality),
        min_chars=max(1, args.min_chars),
        chunk_size=max(300, args.chunk_size),
        overlap=max(20, args.chunk_overlap),
    )
    if not chunks:
        raise RuntimeError("no chunks after filtering; lower thresholds")

    chunks_path = out_dir / "chunks_clean.jsonl"
    write_jsonl(chunks_path, chunks)

    device = args.device
    if device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    texts = [c["text"] for c in chunks]
    emb = embed_chunks(
        model_name=args.embed_model,
        texts=texts,
        batch_size=max(1, args.batch_size),
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
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

