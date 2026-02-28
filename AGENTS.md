# AGENTS.md (pharm-textbook-llm)

Scope: applies to the entire `C:\dev\pharm-textbook-llm` tree.

## 1) Mission

Maintain a stable local pipeline for:

1. OCR/text corpus build from source PDFs
2. Retrieval index build
3. 14B QLoRA finetuning with resume checkpoints
4. One-shot QA testing

## 2) Authoritative Root

- Project root: `C:\dev\pharm-textbook-llm`
- Do not assume `C:\` itself is the project.

## 3) Runtime Safety Rules

1. Do not run destructive commands (`git reset --hard`, mass delete).
2. Do not rename/move project folder while training is active.
3. Prefer `nvidia-smi` for GPU validation over Task Manager.
4. Keep wrapper scripts path-portable (`$PSScriptRoot` style).

## 4) Source-of-Truth Entry Scripts

- `run_train.ps1`
  - `scripts\v3_build_corpus_ocr.py`
  - `scripts\v3_train_knowledge.py`
- `run_finetune.ps1`
  - `scripts\v3_train_lora_qlora.py`
- `run_test.ps1`
  - `scripts\v3_test_one_shot.py`
- `run_test_finetuned.ps1`
  - Resolves latest adapter checkpoint then calls `run_test.ps1`

## 5) Resume/Checkpoint Contract

Any change must preserve:

1. OCR resume state:
   - `work\ocr_v3\state.json`
   - `work\ocr_v3\checkpoints\*.json`
2. Index artifacts:
   - `data\v3_index\chunks_clean.jsonl`
   - `data\v3_index\embeddings.npy`
3. Finetune checkpoints:
   - `data\train_lora_qwen_qwen2_5_14b_v3\checkpoint-*`

## 6) Canonical Commands

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_train.ps1"
```

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_finetune.ps1"
```

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_test.ps1" -Question "?щ????쎈Ъ移섎즺 1李??좏깮??"
```

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_test_finetuned.ps1" -Question "?щ????쎈Ъ移섎즺 1李??좏깮??"
```

## 7) Documentation Sync Rule

If script behavior changes, update in same task:

1. `README.md`
2. `RUNBOOK.md`
3. `AGENTS.md`

## 8) Git/Backup Policy

Commit code/docs only:

- `scripts/`
- `*.ps1`
- `README.md`
- `RUNBOOK.md`
- `AGENTS.md`
- `.gitignore`

Do not commit generated/private artifacts:

- `.venv/`
- `data/`
- `work/`
- `logs/`

