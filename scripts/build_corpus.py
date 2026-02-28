#!/usr/bin/env python3
"""
Full text corpus builder for pharm documents.

Features:
- No OCR. Uses existing text layers / document text only.
- PDF layout-aware extraction with two-column reconstruction heuristic.
- Text normalization for noisy OCR-like spacing artifacts.
- Header/footer repetition removal.
- Resume support with per-file checkpoint state.
- Corpus + RAG chunks + quality reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".hwp"}


@dataclass
class PageRecord:
    doc_id: str
    source_file: str
    page: int
    text: str
    layout_type: str
    quality_score: float
    char_count: int


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stable_doc_id(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8", errors="ignore")).hexdigest()[:16]


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8"):
        pass


def list_target_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    files.sort(key=lambda x: str(x).lower())
    return files


def normalize_hangul_spaced_runs(text: str) -> str:
    # Collapse sequences like "약 물 치 료 학" into "약물치료학".
    pattern = re.compile(r"(?:[가-힣]\s+){2,}[가-힣]")
    prev = None
    out = text
    while prev != out:
        prev = out
        out = pattern.sub(lambda m: m.group(0).replace(" ", ""), out)
    return out


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u00ad", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"([A-Za-z0-9가-힣])-\n([A-Za-z0-9가-힣])", r"\1\2", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = normalize_hangul_spaced_runs(t)
    # Remove lines that are only page numbers / separators.
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if re.fullmatch(r"[-_=~\s\d]{1,20}", s):
            continue
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def score_text_quality(text: str) -> float:
    if not text:
        return 0.0
    char_count = len(text.replace("\n", "").strip())
    if char_count == 0:
        return 0.0
    spaced_hangul = len(re.findall(r"(?:[가-힣]\s+){2,}[가-힣]", text))
    weird_symbol = len(re.findall(r"[�□■◆◇▣◈◉]", text))
    line_count = max(1, text.count("\n") + 1)
    avg_line_len = max(1.0, char_count / line_count)

    score = 1.0
    if char_count < 80:
        score -= 0.25
    if avg_line_len < 8:
        score -= 0.2
    score -= min(0.35, spaced_hangul * 0.02)
    score -= min(0.2, weird_symbol * 0.02)
    return max(0.0, min(1.0, score))


def normalize_repeat_line(line: str) -> str:
    x = re.sub(r"\s+", "", line.strip())
    x = re.sub(r"\d+", "#", x)
    return x[:80]


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

    threshold = max(4, int(len(records) * 0.2))
    frequent_heads = {k for k, v in head_counter.items() if v >= threshold}
    frequent_foots = {k for k, v in foot_counter.items() if v >= threshold}

    out: List[PageRecord] = []
    for r, lines in zip(records, page_lines):
        if lines and normalize_repeat_line(lines[0]) in frequent_heads:
            lines = lines[1:]
        if lines and normalize_repeat_line(lines[-1]) in frequent_foots:
            lines = lines[:-1]
        new_text = "\n".join(lines).strip()
        out.append(
            PageRecord(
                doc_id=r.doc_id,
                source_file=r.source_file,
                page=r.page,
                text=normalize_text(new_text),
                layout_type=r.layout_type,
                quality_score=score_text_quality(new_text),
                char_count=len(new_text.replace("\n", "")),
            )
        )
    return out


def is_double_column(blocks: List[Tuple[float, float, float, float, str]], page_width: float) -> bool:
    if len(blocks) < 4:
        return False
    mid = page_width / 2.0
    left, right = 0, 0
    left_xmax = []
    right_xmin = []
    for x0, y0, x1, y1, txt in blocks:
        if len(txt.strip()) < 8:
            continue
        cx = (x0 + x1) / 2.0
        if cx < mid:
            left += 1
            left_xmax.append(x1)
        else:
            right += 1
            right_xmin.append(x0)
    if left < 2 or right < 2:
        return False
    if not left_xmax or not right_xmin:
        return False
    return (sum(left_xmax) / len(left_xmax)) < (sum(right_xmin) / len(right_xmin))


def extract_pdf_page_text_candidates(
    pdf_doc: fitz.Document,
    pypdf_reader: PdfReader | None,
    page_index: int,
) -> Tuple[List[Tuple[str, str]], int]:
    page = pdf_doc.load_page(page_index)
    page_rect = page.rect
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

    # Candidate A: block-order with optional 2-column handling.
    if blocks:
        if is_double_column(blocks, page_rect.width):
            left = [b for b in blocks if ((b[0] + b[2]) / 2.0) < (page_rect.width / 2.0)]
            right = [b for b in blocks if ((b[0] + b[2]) / 2.0) >= (page_rect.width / 2.0)]
            left.sort(key=lambda x: (round(x[1], 1), round(x[0], 1)))
            right.sort(key=lambda x: (round(x[1], 1), round(x[0], 1)))
            text_a = "\n".join([b[4] for b in left + right])
            candidates.append((text_a, "double_col"))
        else:
            blocks_sorted = sorted(blocks, key=lambda x: (round(x[1], 1), round(x[0], 1)))
            text_a = "\n".join([b[4] for b in blocks_sorted])
            candidates.append((text_a, "single_col"))

    # Candidate B: PyMuPDF plain sorted text.
    text_b = page.get_text("text", sort=True) or ""
    if text_b.strip():
        candidates.append((text_b, "plain_sorted"))

    # Candidate C: pypdf extraction fallback.
    if pypdf_reader is not None:
        try:
            text_c = pypdf_reader.pages[page_index].extract_text() or ""
            if text_c.strip():
                candidates.append((text_c, "pypdf"))
        except Exception:
            pass

    return candidates, len(blocks)


def choose_best_candidate(candidates: List[Tuple[str, str]]) -> Tuple[str, str, float]:
    if not candidates:
        return "", "empty", 0.0
    best_text = ""
    best_layout = "unknown"
    best_score = -1.0
    for text, layout in candidates:
        cleaned = normalize_text(text)
        sc = score_text_quality(cleaned)
        if sc > best_score:
            best_score = sc
            best_text = cleaned
            best_layout = layout
    return best_text, best_layout, max(0.0, best_score)


def process_pdf(
    file_path: Path,
    rel_path: str,
    doc_id: str,
    work_dir: Path,
    flush_every_pages: int,
) -> Tuple[List[PageRecord], str, int, str]:
    """
    Returns: records, status, page_count, error_message
    """
    checkpoints_dir = work_dir / "checkpoints"
    raw_dir = work_dir / "raw_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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

    pypdf_reader = None
    try:
        pypdf_reader = PdfReader(str(file_path))
    except Exception:
        pypdf_reader = None

    if next_page == 0:
        clear_file(raw_path)

    try:
        for i in range(next_page, page_count):
            candidates, _ = extract_pdf_page_text_candidates(pdf_doc, pypdf_reader, i)
            text, layout_type, q = choose_best_candidate(candidates)

            record = {
                "doc_id": doc_id,
                "source_file": rel_path,
                "page": i + 1,
                "text": text,
                "layout_type": layout_type,
                "quality_score": q,
                "char_count": len(text.replace("\n", "")),
            }
            append_jsonl(raw_path, record)

            if (i + 1) % flush_every_pages == 0 or (i + 1) == page_count:
                cp = {
                    "next_page": i + 1,
                    "page_count": page_count,
                    "updated_at": now(),
                }
                write_json(cp_path, cp)

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

    # Post-process full document for repeated headers/footers.
    page_records: List[PageRecord] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            page_records.append(
                PageRecord(
                    doc_id=x["doc_id"],
                    source_file=x["source_file"],
                    page=int(x["page"]),
                    text=x["text"],
                    layout_type=x["layout_type"],
                    quality_score=float(x["quality_score"]),
                    char_count=int(x["char_count"]),
                )
            )

    page_records = remove_repeated_header_footer(page_records)
    cp = {"next_page": page_count, "page_count": page_count, "updated_at": now(), "done": True}
    write_json(cp_path, cp)
    return page_records, "done", page_count, ""


def extract_docx_text(path: Path) -> str:
    doc = Document(str(path))
    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines)


def extract_txt_text(path: Path) -> str:
    # Try utf-8 first, then cp949.
    for enc in ("utf-8", "cp949", "euc-kr", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    with path.open("rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def extract_hwp_text(path: Path) -> str:
    # Prefer hwp5txt if installed.
    cmd = ["hwp5txt", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    except Exception:
        pass
    raise RuntimeError("hwp extraction unavailable (hwp5txt not found or failed)")


def process_non_pdf(path: Path, rel_path: str, doc_id: str) -> Tuple[List[PageRecord], str, int, str]:
    ext = path.suffix.lower()
    try:
        if ext == ".docx":
            raw = extract_docx_text(path)
        elif ext == ".txt":
            raw = extract_txt_text(path)
        elif ext == ".hwp":
            raw = extract_hwp_text(path)
        else:
            return [], "failed", 0, f"unsupported ext: {ext}"
        text = normalize_text(raw)
        rec = PageRecord(
            doc_id=doc_id,
            source_file=rel_path,
            page=1,
            text=text,
            layout_type="non_pdf",
            quality_score=score_text_quality(text),
            char_count=len(text.replace("\n", "")),
        )
        return [rec], "done", 1, ""
    except Exception as e:
        return [], "failed", 0, str(e)


def write_per_doc_records(per_doc_dir: Path, doc_id: str, records: List[PageRecord]):
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
                "quality_score": round(r.quality_score, 6),
                "char_count": r.char_count,
            },
        )


def build_chunks_for_text(text: str, chunk_size: int = 1000, overlap: int = 120) -> List[str]:
    if not text:
        return []
    out = []
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


def merge_outputs(data_dir: Path, manifest_rows: List[dict], chunk_size: int, overlap: int):
    per_doc_dir = data_dir / "per_doc"
    corpus_jsonl = data_dir / "corpus_master.jsonl"
    chunks_jsonl = data_dir / "chunks_rag.jsonl"
    corpus_txt = data_dir / "corpus_master.txt"
    low_quality_csv = data_dir / "low_quality_pages.csv"
    failed_files_csv = data_dir / "failed_files.csv"
    failed_pages_csv = data_dir / "failed_pages.csv"

    clear_file(corpus_jsonl)
    clear_file(chunks_jsonl)
    clear_file(corpus_txt)

    low_quality_rows = []
    failed_file_rows = []
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
                    tf.write(
                        f"[{rec['source_file']} | p.{rec['page']}]\n{rec['text']}\n\n"
                    )
                all_records += 1

                if rec["quality_score"] < 0.55 or rec["char_count"] < 80:
                    low_quality_rows.append(
                        {
                            "doc_id": rec["doc_id"],
                            "source_file": rec["source_file"],
                            "page": rec["page"],
                            "quality_score": rec["quality_score"],
                            "char_count": rec["char_count"],
                            "layout_type": rec["layout_type"],
                        }
                    )

                chunks = build_chunks_for_text(rec["text"], chunk_size=chunk_size, overlap=overlap)
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

    with failed_pages_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc_id", "source_file", "page", "error"]
        )
        writer.writeheader()

    return {
        "total_page_records": all_records,
        "total_chunks": all_chunks,
        "low_quality_pages": len(low_quality_rows),
        "failed_files": len(failed_file_rows),
    }


def write_manifest_csv(path: Path, rows: List[dict]):
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
    resume_cmd: str,
):
    total = len(manifest_rows)
    done = sum(1 for r in manifest_rows if r["status"] == "done")
    failed = sum(1 for r in manifest_rows if r["status"] == "failed")
    pending = total - done - failed
    total_pages = sum(int(r.get("page_count") or 0) for r in manifest_rows)
    elapsed = max(1.0, end_ts - start_ts)
    pps = total_pages / elapsed
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"재실행 명령: {resume_cmd}\n\n")
        f.write("# Run Report\n")
        f.write(f"- started_at: {datetime.fromtimestamp(start_ts)}\n")
        f.write(f"- ended_at: {datetime.fromtimestamp(end_ts)}\n")
        f.write(f"- input_root: {input_root}\n")
        f.write(f"- output_root: {output_root}\n")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every-pages", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    work_dir = output_root / "work"
    data_dir = output_root / "data"
    logs_dir = output_root / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "per_doc").mkdir(parents=True, exist_ok=True)

    state_path = work_dir / "state.json"
    manifest_path = data_dir / "processing_manifest.csv"
    report_path = data_dir / "run_report.md"

    start_ts = time.time()
    all_files = list_target_files(input_root)
    if not all_files:
        print("No target files found.")
        return 1

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

    rows.sort(key=lambda r: r["source_file"].lower())

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

        if row["ext"] == ".pdf":
            records, status, page_count, err = process_pdf(
                full,
                rel,
                row["doc_id"],
                work_dir,
                flush_every_pages=max(1, args.flush_every_pages),
            )
        else:
            records, status, page_count, err = process_non_pdf(full, rel, row["doc_id"])

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
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )

    end_ts = time.time()
    py = Path(sys.executable)
    resume_cmd = (
        f'"{py}" "{Path(__file__).resolve()}" '
        f'--input-root "{input_root}" --output-root "{output_root}" --resume'
    )
    generate_report(
        report_path=report_path,
        input_root=input_root,
        output_root=output_root,
        start_ts=start_ts,
        end_ts=end_ts,
        manifest_rows=rows,
        merge_stats=merge_stats,
        resume_cmd=resume_cmd,
    )
    print("Done.")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

