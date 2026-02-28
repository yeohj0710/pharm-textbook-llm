$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$ProjectRoot = $PSScriptRoot
$Py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "scripts\v3_train_lora_qlora.py"
$Chunks = Join-Path $ProjectRoot "data\v3_index\chunks_clean.jsonl"
$OutDir = Join-Path $ProjectRoot "data\train_lora_qwen_qwen2_5_14b_v3"
$Log = Join-Path $ProjectRoot "logs\finetune_lora.log"

$ModelName = if ($env:PHARM_FT_MODEL_NAME) { $env:PHARM_FT_MODEL_NAME } else { "Qwen/Qwen2.5-14B-Instruct" }
$MaxLength = if ($env:PHARM_FT_MAX_LENGTH) { [int]$env:PHARM_FT_MAX_LENGTH } else { 768 }
$BatchSize = if ($env:PHARM_FT_BATCH_SIZE) { [int]$env:PHARM_FT_BATCH_SIZE } else { 1 }
$GradAccum = if ($env:PHARM_FT_GRAD_ACCUM) { [int]$env:PHARM_FT_GRAD_ACCUM } else { 16 }
$MaxSteps = if ($env:PHARM_FT_MAX_STEPS) { [int]$env:PHARM_FT_MAX_STEPS } else { 20000 }
$SaveSteps = if ($env:PHARM_FT_SAVE_STEPS) { [int]$env:PHARM_FT_SAVE_STEPS } else { 100 }
$LogSteps = if ($env:PHARM_FT_LOGGING_STEPS) { [int]$env:PHARM_FT_LOGGING_STEPS } else { 10 }
$DataWorkers = if ($env:PHARM_FT_DATALOADER_WORKERS) { [int]$env:PHARM_FT_DATALOADER_WORKERS } else { 0 }

if (!(Test-Path $Py)) { throw "python venv not found: $Py" }
if (!(Test-Path $Script)) { throw "script not found: $Script" }
if (!(Test-Path $Chunks)) { throw "chunks file not found: $Chunks (run run_train.ps1 first)" }

New-Item -ItemType Directory -Path (Join-Path $ProjectRoot "logs") -Force | Out-Null
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

"[$(Get-Date)] install finetune deps" | Tee-Object -FilePath $Log -Append
& $Py -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath $Log -Append
& $Py -m pip install --upgrade transformers datasets peft accelerate bitsandbytes sentencepiece 2>&1 | Tee-Object -FilePath $Log -Append

"[$(Get-Date)] start qlora finetune model=$ModelName out_dir=$OutDir max_len=$MaxLength batch=$BatchSize grad_accum=$GradAccum max_steps=$MaxSteps" | Tee-Object -FilePath $Log -Append
$prevErr = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Py $Script `
    --chunks-path $Chunks `
    --output-dir $OutDir `
    --model-name $ModelName `
    --max-length $MaxLength `
    --max-steps $MaxSteps `
    --save-steps $SaveSteps `
    --logging-steps $LogSteps `
    --batch-size $BatchSize `
    --grad-accum $GradAccum `
    --learning-rate 1.2e-4 `
    --warmup-steps 120 `
    --dataloader-workers $DataWorkers `
    --lora-r 16 `
    --lora-alpha 32 `
    --lora-dropout 0.05 `
    --save-total-limit 24 `
    --cpu-offload `
    --gpu-memory-gib 11 `
    --resume 2>&1 | Tee-Object -FilePath $Log -Append
$ftExit = $LASTEXITCODE
$ErrorActionPreference = $prevErr
if ($ftExit -ne 0) {
    throw "QLoRA fine-tuning failed. exit_code=$ftExit"
}
"[$(Get-Date)] end qlora finetune" | Tee-Object -FilePath $Log -Append

