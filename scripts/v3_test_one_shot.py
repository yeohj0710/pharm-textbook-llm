#!/usr/bin/env python3
"""
v3 one-shot QA (no multi-turn memory)

- Hybrid retrieval: dense + BM25
- Single question -> single grounded answer
- Supports optional LoRA adapter and 4bit CPU-offload fallback
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def tok3(text: str) -> List[str]:
    s = re.sub(r"\s+", "", text.lower())
    if not s:
        return []
    if len(s) < 3:
        return [s]
    return [s[i : i + 3] for i in range(len(s) - 2)]


def minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def load_generator(
    model_name: str,
    device: str,
    load_in_4bit: bool,
    cpu_offload: bool,
    gpu_memory_gib: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.float16
        kwargs["attn_implementation"] = "sdpa"
        kwargs["max_memory"] = {0: f"{max(6, int(gpu_memory_gib))}GiB", "cpu": "96GiB"}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=bool(cpu_offload),
            )
    else:
        kwargs["device_map"] = "cpu"
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def build_context(
    rows: List[Dict[str, Any]],
    ranked_idx: List[int],
    max_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = []
    refs: List[Dict[str, Any]] = []
    total = 0
    seen = set()
    for i in ranked_idx:
        r = rows[i]
        source = str(r.get("source_file", ""))
        page = int(r.get("page_start", 0) or 0)
        key = f"{source}::{page}"
        if key in seen:
            continue
        seen.add(key)

        t = normalize_text(str(r.get("text", "")))
        if not t:
            continue
        line = f"[{source} p.{page}] {t}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
        refs.append(
            {
                "source_file": source,
                "page_start": page,
                "page_end": int(r.get("page_end", page) or page),
                "chunk_id": str(r.get("chunk_id", "")),
            }
        )
    return "\n".join(lines), refs


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter-dir", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--gpu-memory-gib", type=int, default=11)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--dense-weight", type=float, default=0.65)
    ap.add_argument("--bm25-weight", type=float, default=0.35)
    ap.add_argument("--max-context-chars", type=int, default=9000)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--plain", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    chunks_path = index_dir / "chunks_clean.jsonl"
    emb_path = index_dir / "embeddings.npy"
    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError("index files missing; run training first")

    rows = read_jsonl(chunks_path)
    if not rows:
        raise RuntimeError("chunks_clean is empty")

    emb = np.load(str(emb_path)).astype(np.float32)
    if emb.shape[0] != len(rows):
        raise RuntimeError(f"embedding/chunk mismatch: {emb.shape[0]} vs {len(rows)}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    em = SentenceTransformer(args.embed_model, device=device if device != "auto" else "cpu")
    qvec = em.encode(
        [f"query: {args.question}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)
    dense_scores = emb @ qvec

    bm25 = BM25Okapi([tok3(str(r.get("text", ""))) for r in rows])
    bm25_scores = np.array(bm25.get_scores(tok3(args.question)), dtype=np.float32)

    dw = float(args.dense_weight)
    bw = float(args.bm25_weight)
    if dw + bw <= 0:
        dw, bw = 0.65, 0.35

    dense_n = minmax(dense_scores)
    bm25_n = minmax(bm25_scores)
    final_scores = dw * dense_n + bw * bm25_n
    rank_idx = np.argsort(-final_scores)[: max(1, args.topk * 3)].tolist()

    context, refs = build_context(rows, rank_idx, max_chars=max(1000, args.max_context_chars))
    if not context:
        raise RuntimeError("empty context after retrieval")

    tokenizer, model = load_generator(
        model_name=args.model_name,
        device=device,
        load_in_4bit=bool(args.load_in_4bit),
        cpu_offload=bool(args.cpu_offload),
        gpu_memory_gib=int(args.gpu_memory_gib),
    )

    if str(args.adapter_dir or "").strip():
        from peft import PeftModel

        adir = Path(str(args.adapter_dir))
        if not adir.exists():
            raise FileNotFoundError(f"adapter dir not found: {adir}")
        model = PeftModel.from_pretrained(model, str(adir))
        model.eval()

    prompt = (
        "You are a pharmacotherapy QA assistant.\n"
        "Rules:\n"
        "1) Use only facts grounded in the provided context.\n"
        "2) If evidence is insufficient, say evidence is insufficient.\n"
        "3) Answer in Korean, 2 to 5 sentences.\n"
        "4) Last line must include citations in format [file p.page].\n"
        "5) No guessing.\n\n"
        f"Question: {args.question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max(32, int(args.max_new_tokens)),
            do_sample=False,
            repetition_penalty=1.03,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    if args.plain:
        print(answer)
        return 0

    print(
        json.dumps(
            {
                "question": args.question,
                "answer": answer,
                "model": args.model_name,
                "device": device,
                "top_refs": refs[: args.topk],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
