#!/usr/bin/env python3
"""
v3 QLoRA fine-tuning (single GPU, resume-safe)

Designed for:
- Qwen2.5-14B-Instruct (4bit QLoRA)
- One-turn QA generation quality improvement
- Resume from latest Trainer checkpoint
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def read_jsonl_texts(path: Path, min_chars: int) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = str(obj.get("text", "") or "").strip()
            if not t:
                continue
            t = re.sub(r"\s+", " ", t)
            if len(t) >= min_chars:
                texts.append(t)
    return texts


def can_use_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(8, device="cuda")
        _ = (x * 2).mean().item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    ck = get_last_checkpoint(str(output_dir))
    return ck


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1.2e-4)
    ap.add_argument("--warmup-steps", type=int, default=120)
    ap.add_argument("--dataloader-workers", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-total-limit", type=int, default=24)
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--gpu-memory-gib", type=int, default=11)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    chunks_path = Path(args.chunks_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")
    if not can_use_cuda():
        raise RuntimeError("CUDA unavailable. This script requires NVIDIA GPU.")

    texts = read_jsonl_texts(chunks_path, min_chars=max(1, int(args.min_chars)))
    if len(texts) < 200:
        raise RuntimeError(f"not enough training texts: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(args.max_length),
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        flat_ids: List[int] = []
        flat_mask: List[int] = []
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            flat_ids.extend(ids)
            flat_mask.extend(mask)
        block = int(args.max_length)
        total = (len(flat_ids) // block) * block
        if total <= 0:
            return {"input_ids": [], "attention_mask": []}
        return {
            "input_ids": [flat_ids[i : i + block] for i in range(0, total, block)],
            "attention_mask": [flat_mask[i : i + block] for i in range(0, total, block)],
        }

    packed = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1024,
        remove_columns=tokenized.column_names,
    )
    packed = packed.filter(lambda x: len(x["input_ids"]) > 32)
    if len(packed) < 200:
        raise RuntimeError(f"tokenized dataset too small after packing: {len(packed)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=bool(args.cpu_offload),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory={0: f"{max(6, int(args.gpu_memory_gib))}GiB", "cpu": "96GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=max(4, int(args.lora_r)),
        lora_alpha=max(8, int(args.lora_alpha)),
        lora_dropout=float(args.lora_dropout),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        per_device_train_batch_size=max(1, int(args.batch_size)),
        gradient_accumulation_steps=max(1, int(args.grad_accum)),
        learning_rate=float(args.learning_rate),
        max_steps=max(1, int(args.max_steps)),
        warmup_steps=max(0, int(args.warmup_steps)),
        logging_steps=max(1, int(args.logging_steps)),
        save_steps=max(1, int(args.save_steps)),
        save_total_limit=max(1, int(args.save_total_limit)),
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        dataloader_num_workers=max(0, int(args.dataloader_workers)),
        dataloader_pin_memory=True,
        report_to=[],
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        tf32=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=packed,
        data_collator=collator,
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(output_dir)

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_ckpt)
    t1 = time.time()

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": args.model_name,
        "chunks_path": str(chunks_path),
        "output_dir": str(output_dir),
        "final_adapter_dir": str(final_dir),
        "device_name": torch.cuda.get_device_name(0),
        "train_samples": len(packed),
        "max_steps": int(args.max_steps),
        "elapsed_sec": round(t1 - t0, 2),
        "resume_from_checkpoint": resume_ckpt or "",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
