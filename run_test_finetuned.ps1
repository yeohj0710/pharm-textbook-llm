param(
    [Parameter(Mandatory = $true)]
    [string]$Question,
    [ValidateSet("auto","cpu","cuda")]
    [string]$Device = "cuda",
    [string]$ModelName = "Qwen/Qwen2.5-14B-Instruct"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$RunTest = Join-Path $ProjectRoot "run_test.ps1"
$TrainDir = Join-Path $ProjectRoot "data\train_lora_qwen_qwen2_5_14b_v3"

if (!(Test-Path $RunTest)) { throw "run_test.ps1 not found: $RunTest" }
if (!(Test-Path $TrainDir)) { throw "train dir not found: $TrainDir (run run_finetune.ps1 first)" }

$candidates = Get-ChildItem -Path $TrainDir -Directory -Filter "checkpoint-*" | Sort-Object Name -Descending
$adapter = $null
foreach ($c in $candidates) {
    $cfg = Join-Path $c.FullName "adapter_config.json"
    if (Test-Path $cfg) {
        $adapter = $c.FullName
        break
    }
}
if (-not $adapter) {
    $final = Join-Path $TrainDir "final_adapter"
    if (Test-Path (Join-Path $final "adapter_config.json")) {
        $adapter = $final
    }
}
if (-not $adapter) {
    throw "no adapter found in $TrainDir"
}

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $RunTest `
    -Question $Question `
    -ModelName $ModelName `
    -Device $Device `
    -AdapterDir $adapter

