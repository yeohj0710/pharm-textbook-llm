#!/usr/bin/env python3
"""
v3 OCR-first corpus builder for scanned pharmacotherapy PDFs.

Goals:
- Robustly process scanned PDF pages with two-column body text.
- Keep resume/checkpoint safety per document/page.
- Preserve page-level provenance for downstream QA citations.

Pipeline:
1) text-layer extraction candidates (PyMuPDF / pypdf)
2) quality scoring
3) low-quality pages -> image OCR with two-column split heuristic
4) per-doc postprocess (repeated header/footer trimming)
5) export corpus/chunks/quality reports
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np
from pypdf import PdfReader

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


SUPPORTED_EXTS = {".pdf"}


@dataclass
class PageRecord:
    doc_id: str
    source_file: str
    page: int
    text: str
    layout_type: str
    method: str
    quality_score: float
    char_count: int
    ocr_confidence: float


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stable_doc_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8"):
        pass


def list_target_files(
    root: Path,
    include_regex: str = "",
    exclude_regex: str = "",
) -> List[Path]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    files: List[Path] = []
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTS):
            continue
        name = p.name
        rel = str(p.relative_to(root))
        target = f"{rel}::{name}"
        if inc is not None and not inc.search(target):
            continue
        if exc is not None and exc.search(target):
            continue
        files.append(p)
    files.sort(key=lambda x: str(x).lower())
    return files


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u00ad", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Common OCR join: word-\nword
    t = re.sub(r"([0-9A-Za-z가-힣])-\n([0-9A-Za-z가-힣])", r"\1\2", t)

    # Remove repeated spaces and trim line edge spaces.
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    # Reconnect spaced Hangul runs: "약 물 치 료 학" -> "약물치료학"
    t = re.sub(
        r"(?:[가-힣]\s+){2,}[가-힣]",
        lambda m: re.sub(r"\s+", "", m.group(0)),
        t,
    )

    lines: List[str] = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        # drop tiny separator/page-only lines
        if re.fullmatch(r"[-_=~\s\d]{1,24}", s):
            continue
        lines.append(s)

    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def score_text_quality(text: str) -> float:
    if not text:
        return 0.0
    body = text.replace("\n", "")
    n = len(body)
    if n == 0:
        return 0.0

    hangul = len(re.findall(r"[가-힣]", body))
    latin_num = len(re.findall(r"[A-Za-z0-9]", body))
    weird = len(re.findall(r"[�□■◆◇¤§¶†‡※★☆▽△◁▷▶◀⟂⟪⟫]", body))
    punct = len(re.findall(r"[`~^|{}<>]", body))
    spaced_hangul_runs = len(re.findall(r"(?:[가-힣]\s+){2,}[가-힣]", text))

    line_count = max(1, text.count("\n") + 1)
    avg_line = n / line_count

    score = 1.0
    if n < 120:
        score -= 0.30
    elif n < 220:
        score -= 0.10

    hangul_ratio = hangul / max(1, n)
    if hangul_ratio < 0.20:
        score -= 0.20
    if avg_line < 9:
        score -= 0.18

    score -= min(0.30, spaced_hangul_runs * 0.02)
    score -= min(0.20, (weird + punct) / max(1, n) * 4.0)

    # slight bonus for realistic mixed text.
    if hangul > 0 and latin_num > 0:
        score += 0.03

    return max(0.0, min(1.0, score))


def normalize_repeat_line(line: str) -> str:
    x = re.sub(r"\s+", "", line.strip())
    x = re.sub(r"\d+", "#", x)
    return x[:100]


def remove_repeated_header_footer(records: List[PageRecord]) -> List[PageRecord]:
    if len(records) < 6:
        return records
    head_counter: Counter = Counter()
    foot_counter: Counter = Counter()

    page_lines: List[List[str]] = []
    for r in records:
        lines = [ln.strip() for ln in r.text.split("\n") if ln.strip()]
        page_lines.append(lines)
        if lines:
            head_counter[normalize_repeat_line(lines[0])] += 1
            foot_counter[normalize_repeat_line(lines[-1])] += 1

    threshold = max(4, int(len(records) * 0.20))
    frequent_heads = {k for k, v in head_counter.items() if v >= threshold}
    frequent_foots = {k for k, v in foot_counter.items() if v >= threshold}

    out: List[PageRecord] = []
    for r, lines in zip(records, page_lines):
        if lines and normalize_repeat_line(lines[0]) in frequent_heads:
            lines = lines[1:]
        if lines and normalize_repeat_line(lines[-1]) in frequent_foots:
            lines = lines[:-1]
        new_text = normalize_text("\n".join(lines))
        out.append(
            PageRecord(
                doc_id=r.doc_id,
                source_file=r.source_file,
                page=r.page,
                text=new_text,
                layout_type=r.layout_type,
                method=r.method,
                quality_score=score_text_quality(new_text),
                char_count=len(new_text.replace("\n", "")),
                ocr_confidence=r.ocr_confidence,
            )
        )
    return out


def is_double_column_blocks(
    blocks: List[Tuple[float, float, float, float, str]],
    page_width: float,
) -> bool:
    if len(blocks) < 4:
        return False
    mid = page_width / 2.0
    left, right = 0, 0
    left_xmax: List[float] = []
    right_xmin: List[float] = []
    for x0, y0, x1, y1, txt in blocks:
        if len(txt.strip()) < 10:
            continue
        cx = (x0 + x1) / 2.0
        if cx < mid:
            left += 1
            left_xmax.append(x1)
        else:
            right += 1
            right_xmin.append(x0)
    if left < 2 or right < 2 or not left_xmax or not right_xmin:
        return False
    return (sum(left_xmax) / len(left_xmax)) < (sum(right_xmin) / len(right_xmin))


def extract_text_layer_candidates(
    pdf_doc: fitz.Document,
    pypdf_reader: Optional[PdfReader],
    page_index: int,
) -> List[Tuple[str, str]]:
    page = pdf_doc.load_page(page_index)
    rect = page.rect
    blocks_raw = page.get_text("blocks")
    blocks: List[Tuple[float, float, float, float, str]] = []
    for b in blocks_raw:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, txt = b[:5]
        txt = (txt or "").strip()
        if txt:
            blocks.append((float(x0), float(y0), float(x1), float(y1), txt))

    candidates: List[Tuple[str, str]] = []

    if blocks:
        if is_double_column_blocks(blocks, rect.width):
            left = [b for b in blocks if ((b[0] + b[2]) / 2.0) < (rect.width / 2.0)]
            right = [b for b in blocks if ((b[0] + b[2]) / 2.0) >= (rect.width / 2.0)]
            left.sort(key=lambda x: (round(x[1], 1), round(x[0], 1)))
            right.sort(key=lambda x: (round(x[1], 1), round(x[0], 1)))
            txt = "\n".join([x[4] for x in left + right])
            candidates.append((txt, "textlayer_double_col"))
        else:
            blocks_sorted = sorted(blocks, key=lambda x: (round(x[1], 1), round(x[0], 1)))
            txt = "\n".join([x[4] for x in blocks_sorted])
            candidates.append((txt, "textlayer_block"))

    plain = page.get_text("text", sort=True) or ""
    if plain.strip():
        candidates.append((plain, "textlayer_plain"))

    if pypdf_reader is not None:
        try:
            t = pypdf_reader.pages[page_index].extract_text() or ""
            if t.strip():
                candidates.append((t, "textlayer_pypdf"))
        except Exception:
            pass

    return candidates


def choose_best_text_candidate(candidates: List[Tuple[str, str]]) -> Tuple[str, str, float]:
    if not candidates:
        return "", "empty", 0.0
    best_text = ""
    best_layout = "unknown"
    best_score = -1.0
    for txt, layout in candidates:
        cleaned = normalize_text(txt)
        sc = score_text_quality(cleaned)
        if sc > best_score:
            best_score = sc
            best_text = cleaned
            best_layout = layout
    return best_text, best_layout, max(0.0, best_score)


class OCREngine:
    def __init__(self) -> None:
        self._rapid = None
        self._rapid_name = ""
        self._init_rapid()

    def _init_rapid(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._rapid = RapidOCR()
            self._rapid_name = "rapidocr_onnxruntime"
        except Exception:
            self._rapid = None
            self._rapid_name = ""

    @property
    def name(self) -> str:
        return self._rapid_name or "none"

    def available(self) -> bool:
        return self._rapid is not None

    def run(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        if self._rapid is None:
            return "", 0.0
        try:
            result, _ = self._rapid(image_bgr)
        except Exception:
            return "", 0.0
        if not result:
            return "", 0.0

        # result: [ [box, text, score], ... ]
        rows: List[Tuple[float, float, str, float]] = []
        for item in result:
            if not item or len(item) < 3:
                continue
            box, text, score = item[0], str(item[1]), float(item[2])
            if not text.strip():
                continue
            try:
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
                x0 = min(xs)
                y0 = min(ys)
            except Exception:
                x0, y0 = 0.0, 0.0
            rows.append((y0, x0, text.strip(), score))

        if not rows:
            return "", 0.0

        rows.sort(key=lambda x: (round(x[0], 1), round(x[1], 1)))
        text = "\n".join(x[2] for x in rows)
        conf = float(sum(x[3] for x in rows) / len(rows))
        return normalize_text(text), conf


def pixmap_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    scale = max(1.0, float(dpi) / 72.0)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if arr.shape[2] == 4 and cv2 is not None:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if arr.shape[2] == 3 and cv2 is not None:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # no-op fallback
    return arr[:, :, :3].copy()


def detect_column_split_x(image_bgr: np.ndarray) -> Optional[int]:
    if cv2 is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu to get text mask
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = (bw > 0).mean(axis=0).astype(np.float32)
    if density.size < 20:
        return None

    kernel = np.ones(31, dtype=np.float32) / 31.0
    smooth = np.convolve(density, kernel, mode="same")
    w = smooth.shape[0]
    c0 = int(w * 0.30)
    c1 = int(w * 0.70)
    if c1 <= c0 + 10:
        return None
    mid_slice = smooth[c0:c1]
    idx_local = int(np.argmin(mid_slice))
    split_x = c0 + idx_local

    valley = float(smooth[split_x])
    left_mean = float(np.mean(smooth[max(0, split_x - int(w * 0.20)):split_x]))
    right_mean = float(np.mean(smooth[split_x:min(w, split_x + int(w * 0.20))]))

    # Require clear valley between two text masses.
    if left_mean < 0.015 or right_mean < 0.015:
        return None
    if valley > min(left_mean, right_mean) * 0.60:
        return None
    return split_x


def extract_ocr_text_from_page(
    page: fitz.Page,
    ocr: OCREngine,
    dpi: int,
) -> Tuple[str, str, float]:
    if not ocr.available():
        return "", "ocr_unavailable", 0.0

    image = pixmap_to_bgr(page, dpi=dpi)
    h, w = image.shape[:2]
    if h < 20 or w < 20:
        return "", "ocr_image_invalid", 0.0

    split_x = detect_column_split_x(image)
    parts: List[np.ndarray] = []
    layout = "ocr_single_col"
    if split_x is not None:
        gutter = max(8, int(w * 0.01))
        lx1 = max(1, split_x - gutter)
        rx0 = min(w - 1, split_x + gutter)
        left = image[:, :lx1]
        right = image[:, rx0:]
        if left.shape[1] > 40 and right.shape[1] > 40:
            parts = [left, right]
            layout = "ocr_double_col"
    if not parts:
        parts = [image]

    texts: List[str] = []
    confs: List[float] = []
    for crop in parts:
        t, c = ocr.run(crop)
        if t.strip():
            texts.append(t)
            confs.append(c)

    if not texts:
        return "", layout, 0.0
    merged = normalize_text("\n".join(texts))
    conf = float(sum(confs) / len(confs)) if confs else 0.0
    return merged, layout, conf


def write_per_doc_records(per_doc_dir: Path, doc_id: str, records: List[PageRecord]) -> None:
    out = per_doc_dir / f"{doc_id}.jsonl"
    clear_file(out)
    for r in records:
        append_jsonl(
            out,
            {
                "doc_id": r.doc_id,
                "source_file": r.source_file,
                "page": r.page,
                "text": r.text,
                "layout_type": r.layout_type,
                "method": r.method,
                "quality_score": round(r.quality_score, 6),
                "char_count": r.char_count,
                "ocr_confidence": round(r.ocr_confidence, 6),
            },
        )


def build_chunks_for_text(text: str, chunk_size: int = 1000, overlap: int = 120) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if end < n:
            last_break = max(chunk.rfind("\n"), chunk.rfind(". "), chunk.rfind(" "))
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


def merge_outputs(
    data_dir: Path,
    manifest_rows: List[dict],
    chunk_size: int,
    overlap: int,
) -> Dict[str, Any]:
    per_doc_dir = data_dir / "per_doc"
    corpus_jsonl = data_dir / "corpus_master.jsonl"
    chunks_jsonl = data_dir / "chunks_rag.jsonl"
    corpus_txt = data_dir / "corpus_master.txt"
    low_quality_csv = data_dir / "low_quality_pages.csv"
    failed_files_csv = data_dir / "failed_files.csv"

    clear_file(corpus_jsonl)
    clear_file(chunks_jsonl)
    clear_file(corpus_txt)

    low_quality_rows: List[Dict[str, Any]] = []
    failed_file_rows: List[Dict[str, Any]] = []
    all_records = 0
    all_chunks = 0

    for row in manifest_rows:
        if row["status"] != "done":
            failed_file_rows.append(row)
            continue
        doc_id = row["doc_id"]
        p = per_doc_dir / f"{doc_id}.jsonl"
        if not p.exists():
            failed = dict(row)
            failed["error"] = "missing_per_doc_output"
            failed_file_rows.append(failed)
            continue

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                append_jsonl(corpus_jsonl, rec)
                with corpus_txt.open("a", encoding="utf-8") as tf:
                    tf.write(f"[{rec['source_file']} | p.{rec['page']}]\n{rec['text']}\n\n")
                all_records += 1

                if float(rec.get("quality_score", 0.0)) < 0.72 or int(rec.get("char_count", 0)) < 120:
                    low_quality_rows.append(
                        {
                            "doc_id": rec.get("doc_id", ""),
                            "source_file": rec.get("source_file", ""),
                            "page": rec.get("page", 0),
                            "quality_score": rec.get("quality_score", 0.0),
                            "char_count": rec.get("char_count", 0),
                            "layout_type": rec.get("layout_type", ""),
                            "method": rec.get("method", ""),
                            "ocr_confidence": rec.get("ocr_confidence", 0.0),
                        }
                    )

                chunks = build_chunks_for_text(
                    str(rec.get("text", "")),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                for idx, chunk in enumerate(chunks, start=1):
                    chunk_rec = {
                        "chunk_id": f"{rec['doc_id']}_p{rec['page']}_c{idx}",
                        "doc_id": rec["doc_id"],
                        "source_file": rec["source_file"],
                        "page_start": rec["page"],
                        "page_end": rec["page"],
                        "text": chunk,
                    }
                    append_jsonl(chunks_jsonl, chunk_rec)
                    all_chunks += 1

    low_quality_rows.sort(key=lambda x: (x["quality_score"], x["char_count"]))
    with low_quality_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "source_file",
                "page",
                "quality_score",
                "char_count",
                "layout_type",
                "method",
                "ocr_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(low_quality_rows)

    with failed_files_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["doc_id", "source_file", "ext", "status", "page_count", "error", "updated_at"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in failed_file_rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    return {
        "total_page_records": all_records,
        "total_chunks": all_chunks,
        "low_quality_pages": len(low_quality_rows),
        "failed_files": len(failed_file_rows),
    }


def process_pdf(
    file_path: Path,
    rel_path: str,
    doc_id: str,
    work_dir: Path,
    flush_every_pages: int,
    quality_threshold: float,
    min_chars_for_textlayer: int,
    dpi: int,
    ocr: OCREngine,
) -> Tuple[List[PageRecord], str, int, str]:
    checkpoints_dir = work_dir / "checkpoints"
    raw_dir = work_dir / "raw_pages"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cp_path = checkpoints_dir / f"{doc_id}.json"
    raw_path = raw_dir / f"{doc_id}.jsonl"
    cp = read_json(cp_path, {"next_page": 0, "page_count": None, "updated_at": None})
    next_page = int(cp.get("next_page", 0))

    try:
        pdf_doc = fitz.open(file_path)
        page_count = len(pdf_doc)
    except Exception as e:
        return [], "failed", 0, f"fitz_open_failed: {e}"

    if cp.get("page_count") is None:
        cp["page_count"] = page_count
    write_json(cp_path, cp)

    pypdf_reader: Optional[PdfReader] = None
    try:
        pypdf_reader = PdfReader(str(file_path))
    except Exception:
        pypdf_reader = None

    if next_page == 0:
        clear_file(raw_path)

    try:
        for i in range(next_page, page_count):
            page = pdf_doc.load_page(i)
            candidates = extract_text_layer_candidates(pdf_doc, pypdf_reader, i)
            txt_base, layout_base, q_base = choose_best_text_candidate(candidates)
            c_base = len(txt_base.replace("\n", ""))

            selected_text = txt_base
            selected_layout = layout_base
            selected_method = "textlayer"
            selected_q = q_base
            selected_conf = 0.0

            need_ocr = (q_base < quality_threshold) or (c_base < min_chars_for_textlayer)
            if need_ocr:
                txt_ocr, layout_ocr, conf_ocr = extract_ocr_text_from_page(page, ocr=ocr, dpi=dpi)
                q_ocr = score_text_quality(txt_ocr)
                c_ocr = len(txt_ocr.replace("\n", ""))
                # Replace with OCR if better quality or clearly more content.
                if txt_ocr and (q_ocr > q_base + 0.05 or c_ocr > int(max(40, c_base * 1.20))):
                    selected_text = txt_ocr
                    selected_layout = layout_ocr
                    selected_method = f"ocr:{ocr.name}"
                    selected_q = q_ocr
                    selected_conf = conf_ocr

            record = {
                "doc_id": doc_id,
                "source_file": rel_path,
                "page": i + 1,
                "text": selected_text,
                "layout_type": selected_layout,
                "method": selected_method,
                "quality_score": selected_q,
                "char_count": len(selected_text.replace("\n", "")),
                "ocr_confidence": selected_conf,
            }
            append_jsonl(raw_path, record)

            if (i + 1) % flush_every_pages == 0 or (i + 1) == page_count:
                cp = {
                    "next_page": i + 1,
                    "page_count": page_count,
                    "updated_at": now(),
                }
                write_json(cp_path, cp)
                print(
                    f"[{now()}] {rel_path} page {i+1}/{page_count} "
                    f"method={selected_method} q={selected_q:.3f}",
                    flush=True,
                )
    except Exception as e:
        cp = {
            "next_page": i if "i" in locals() else next_page,
            "page_count": page_count,
            "updated_at": now(),
            "error": str(e),
        }
        write_json(cp_path, cp)
        return [], "failed", page_count, f"pdf_page_processing_failed: {e}"
    finally:
        pdf_doc.close()

    page_records: List[PageRecord] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            page_records.append(
                PageRecord(
                    doc_id=x["doc_id"],
                    source_file=x["source_file"],
                    page=int(x["page"]),
                    text=str(x.get("text", "")),
                    layout_type=str(x.get("layout_type", "")),
                    method=str(x.get("method", "")),
                    quality_score=float(x.get("quality_score", 0.0) or 0.0),
                    char_count=int(x.get("char_count", 0) or 0),
                    ocr_confidence=float(x.get("ocr_confidence", 0.0) or 0.0),
                )
            )

    page_records = remove_repeated_header_footer(page_records)
    cp = {"next_page": page_count, "page_count": page_count, "updated_at": now(), "done": True}
    write_json(cp_path, cp)
    return page_records, "done", page_count, ""


def write_manifest_csv(path: Path, rows: List[dict]) -> None:
    fields = ["doc_id", "source_file", "ext", "status", "page_count", "error", "updated_at"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def generate_report(
    report_path: Path,
    input_root: Path,
    output_root: Path,
    start_ts: float,
    end_ts: float,
    manifest_rows: List[dict],
    merge_stats: dict,
    ocr_engine_name: str,
    resume_cmd: str,
) -> None:
    total = len(manifest_rows)
    done = sum(1 for r in manifest_rows if r["status"] == "done")
    failed = sum(1 for r in manifest_rows if r["status"] == "failed")
    pending = total - done - failed
    total_pages = sum(int(r.get("page_count") or 0) for r in manifest_rows)
    elapsed = max(1.0, end_ts - start_ts)
    pps = total_pages / elapsed
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# OCR v3 Run Report\n\n")
        f.write(f"- started_at: {datetime.fromtimestamp(start_ts)}\n")
        f.write(f"- ended_at: {datetime.fromtimestamp(end_ts)}\n")
        f.write(f"- input_root: {input_root}\n")
        f.write(f"- output_root: {output_root}\n")
        f.write(f"- ocr_engine: {ocr_engine_name}\n")
        f.write(f"- total_files: {total}\n")
        f.write(f"- done_files: {done}\n")
        f.write(f"- failed_files: {failed}\n")
        f.write(f"- pending_files: {pending}\n")
        f.write(f"- total_pages: {total_pages}\n")
        f.write(f"- avg_pages_per_sec: {pps:.4f}\n")
        f.write(f"- total_page_records: {merge_stats.get('total_page_records', 0)}\n")
        f.write(f"- total_chunks: {merge_stats.get('total_chunks', 0)}\n")
        f.write(f"- low_quality_pages: {merge_stats.get('low_quality_pages', 0)}\n")
        f.write(f"- failed_files_in_merge: {merge_stats.get('failed_files', 0)}\n")
        f.write(f"\n- resume_cmd: `{resume_cmd}`\n")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every-pages", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--dpi", type=int, default=340)
    ap.add_argument("--quality-threshold", type=float, default=0.90)
    ap.add_argument("--min-chars-for-textlayer", type=int, default=180)
    ap.add_argument("--include-regex", default="")
    ap.add_argument("--exclude-regex", default="")
    args = ap.parse_args()

    if cv2 is None:
        raise RuntimeError("opencv-python-headless not installed. Please install requirements first.")

    input_root = Path(args.input_root)
    project_root = Path(args.project_root)
    output_root = project_root

    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    work_dir = output_root / "work" / "ocr_v3"
    data_dir = output_root / "data"
    logs_dir = output_root / "logs"
    (data_dir / "per_doc").mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    state_path = work_dir / "state.json"
    manifest_path = data_dir / "processing_manifest.csv"
    report_path = data_dir / "run_report.md"

    start_ts = time.time()
    all_files = list_target_files(
        input_root,
        include_regex=str(args.include_regex or ""),
        exclude_regex=str(args.exclude_regex or ""),
    )
    if not all_files:
        print("No target PDF files found.")
        return 1

    ocr = OCREngine()
    if not ocr.available():
        raise RuntimeError(
            "No OCR backend available. Install rapidocr-onnxruntime first."
        )
    print(f"[{now()}] OCR engine: {ocr.name}", flush=True)

    existing_state = read_json(state_path, {})
    state_rows = existing_state.get("rows", [])
    state_map: Dict[str, dict] = {r["source_file"]: r for r in state_rows}

    rows: List[dict] = []
    for fp in all_files:
        rel = str(fp.relative_to(input_root))
        ext = fp.suffix.lower()
        doc_id = stable_doc_id(rel)
        prev = state_map.get(rel, {})
        status = prev.get("status", "pending")
        if not args.resume:
            status = "pending"
        rows.append(
            {
                "doc_id": doc_id,
                "source_file": rel,
                "ext": ext,
                "status": status,
                "page_count": prev.get("page_count", 0),
                "error": prev.get("error", ""),
                "updated_at": prev.get("updated_at", ""),
            }
        )
    rows.sort(key=lambda r: str(r["source_file"]).lower())

    for row in rows:
        if args.resume and row["status"] == "done":
            continue

        rel = row["source_file"]
        full = input_root / rel
        row["status"] = "running"
        row["updated_at"] = now()
        row["error"] = ""
        write_json(state_path, {"rows": rows, "updated_at": now()})
        write_manifest_csv(manifest_path, rows)

        records, status, page_count, err = process_pdf(
            full,
            rel,
            row["doc_id"],
            work_dir=work_dir,
            flush_every_pages=max(1, args.flush_every_pages),
            quality_threshold=float(args.quality_threshold),
            min_chars_for_textlayer=max(1, int(args.min_chars_for_textlayer)),
            dpi=max(160, int(args.dpi)),
            ocr=ocr,
        )

        row["status"] = status
        row["page_count"] = page_count
        row["error"] = err
        row["updated_at"] = now()

        if status == "done":
            write_per_doc_records(data_dir / "per_doc", row["doc_id"], records)

        write_json(state_path, {"rows": rows, "updated_at": now()})
        write_manifest_csv(manifest_path, rows)

    merge_stats = merge_outputs(
        data_dir=data_dir,
        manifest_rows=rows,
        chunk_size=max(400, int(args.chunk_size)),
        overlap=max(40, int(args.chunk_overlap)),
    )

    end_ts = time.time()
    py = Path(sys.executable)
    resume_cmd = (
        f'"{py}" "{Path(__file__).resolve()}" '
        f'--input-root "{input_root}" --project-root "{project_root}" --resume'
    )
    generate_report(
        report_path=report_path,
        input_root=input_root,
        output_root=output_root,
        start_ts=start_ts,
        end_ts=end_ts,
        manifest_rows=rows,
        merge_stats=merge_stats,
        ocr_engine_name=ocr.name,
        resume_cmd=resume_cmd,
    )

    print("Done.")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
