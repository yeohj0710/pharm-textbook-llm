# Pharm Textbook LLM Runbook (v3 OCR + One-shot QA)

Single reference for run/resume/test operations.

## 1) Project root

`C:\dev\pharm-textbook-llm`

## 2) Source PDFs

Source root is encoded inside `run_train.ps1`.

Decode in PowerShell:

```powershell
[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("QzpcVXNlcnNcaGp5ZW9cT25lRHJpdmVcRGVza3RvcFxVU0Ig7J6Q66OMXDUtMVzqs7XthrU="))
```

## 3) Build corpus + index (resume-safe)

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_train.ps1"
```

Outputs:

- `data\corpus_master.jsonl`
- `data\v3_index\chunks_clean.jsonl`
- `data\v3_index\embeddings.npy`
- `data\v3_index\meta.json`

## 4) Base one-shot test

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_test.ps1" -Question "?щ????쎈Ъ移섎즺 1李??좏깮??"
```

## 5) Finetune start/resume (14B QLoRA)

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_finetune.ps1"
```

Checkpoint path:

- `data\train_lora_qwen_qwen2_5_14b_v3\checkpoint-*`

## 6) Finetuned test (latest checkpoint auto-select)

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_test_finetuned.ps1" -Question "?щ????쎈Ъ移섎즺 1李??좏깮??"
```

## 7) Resume behavior

If interrupted/rebooted:

1. Re-run `run_train.ps1` to resume OCR build/index stage.
2. Re-run `run_finetune.ps1` to resume from latest trainer checkpoint.
3. Use `run_test_finetuned.ps1` anytime to check latest adapter quality.

## 8) Optional finetune env overrides

Set before running `run_finetune.ps1`:

- `PHARM_FT_MODEL_NAME`
- `PHARM_FT_MAX_LENGTH`
- `PHARM_FT_BATCH_SIZE`
- `PHARM_FT_GRAD_ACCUM`
- `PHARM_FT_MAX_STEPS`
- `PHARM_FT_SAVE_STEPS`
- `PHARM_FT_LOGGING_STEPS`
- `PHARM_FT_DATALOADER_WORKERS`

Example in Bash:

```bash
export PHARM_FT_MAX_LENGTH=384
export PHARM_FT_BATCH_SIZE=1
export PHARM_FT_GRAD_ACCUM=4
export PHARM_FT_MAX_STEPS=800
export PHARM_FT_SAVE_STEPS=50
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\dev\pharm-textbook-llm\run_finetune.ps1"
```

## 9) GPU monitoring

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 1
```

Use this instead of Task Manager 3D/NPU graphs for training validity checks.

