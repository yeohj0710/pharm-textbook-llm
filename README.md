# pharm-textbook-llm

Local one-shot QA pipeline specialized for the pharmacotherapy textbook set.

This repo currently uses a 3-stage flow:

1. OCR/text cleanup + corpus build + retrieval index build
2. 14B QLoRA fine-tuning (resume-safe checkpoints)
3. One-shot question answering (base model or finetuned adapter)

## Project Root

`C:\dev\pharm-textbook-llm`

## Active Entry Scripts

- `run_train.ps1`
  - Builds OCR corpus and `data\v3_index`
- `run_finetune.ps1`
  - Runs/resumes QLoRA training on 14B base
- `run_test.ps1`
  - One-shot test with optional adapter
- `run_test_finetuned.ps1`
  - Auto-picks latest adapter checkpoint and tests

## Canonical Commands

### 1) Build/refresh corpus + retrieval index (resume-safe)

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_train.ps1"
```

### 2) Start/resume finetuning

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_finetune.ps1"
```

### 3) Test base model (one-shot)

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_test.ps1" -Question "?щ????쎈Ъ移섎즺 1李??좏깮??"
```

### 4) Test latest finetuned checkpoint (one-shot)

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_test_finetuned.ps1" -Question "?щ????쎈Ъ移섎즺 1李??좏깮??"
```

## Resume/Checkpoint Paths

- OCR resume state:
  - `work\ocr_v3\state.json`
  - `work\ocr_v3\checkpoints\*.json`
  - `work\ocr_v3\raw_pages\*.jsonl`
- Retrieval index:
  - `data\v3_index\chunks_clean.jsonl`
  - `data\v3_index\embeddings.npy`
  - `data\v3_index\meta.json`
- Finetuning checkpoints:
  - `data\train_lora_qwen_qwen2_5_14b_v3\checkpoint-*`
  - `data\train_lora_qwen_qwen2_5_14b_v3\final_adapter`

## Finetune Environment Overrides

`run_finetune.ps1` supports:

- `PHARM_FT_MODEL_NAME`
- `PHARM_FT_MAX_LENGTH`
- `PHARM_FT_BATCH_SIZE`
- `PHARM_FT_GRAD_ACCUM`
- `PHARM_FT_MAX_STEPS`
- `PHARM_FT_SAVE_STEPS`
- `PHARM_FT_LOGGING_STEPS`
- `PHARM_FT_DATALOADER_WORKERS`

## GPU Check

Prefer `nvidia-smi` over Task Manager:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1
```

## Notes

- This is one-shot QA by design (no multi-turn memory state).
- OCR quality is still the largest quality bottleneck for exact numeric/table facts.
- Generated artifacts are intentionally excluded from git via `.gitignore`.

