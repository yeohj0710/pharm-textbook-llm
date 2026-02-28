param(
    [Parameter(Mandatory = $true)]
    [string]$Question,
    [string]$ModelName = "Qwen/Qwen2.5-14B-Instruct",
    [ValidateSet("auto","cpu","cuda")]
    [string]$Device = "cuda",
    [string]$AdapterDir = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$Py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "scripts\v3_test_one_shot.py"
$IndexDir = Join-Path $ProjectRoot "data\v3_index"

if (!(Test-Path $Py)) {
    throw "python venv not found: $Py"
}
if (!(Test-Path $Script)) {
    throw "script not found: $Script"
}
if (!(Test-Path $IndexDir)) {
    throw "index dir not found: $IndexDir (run training first)"
}

if ($AdapterDir -and !(Test-Path $AdapterDir)) {
    throw "adapter dir not found: $AdapterDir"
}

$cmd = @(
    "--index-dir", $IndexDir,
    "--question", $Question,
    "--embed-model", "intfloat/multilingual-e5-large-instruct",
    "--model-name", $ModelName,
    "--device", $Device,
    "--load-in-4bit",
    "--cpu-offload",
    "--gpu-memory-gib", "11",
    "--topk", "8",
    "--dense-weight", "0.65",
    "--bm25-weight", "0.35",
    "--max-context-chars", "9000",
    "--max-new-tokens", "180",
    "--plain"
)
if ($AdapterDir) {
    $cmd += @("--adapter-dir", $AdapterDir)
}

& $Py $Script @cmd

