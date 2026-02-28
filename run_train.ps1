$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$InputRoot = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("QzpcVXNlcnNcaGp5ZW9cT25lRHJpdmVcRGVza3RvcFxVU0Ig7J6Q66OMXDUtMVzqs7XthrU="))
$IncludeRegex = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("7JW966y87LmY66OM7ZWZIOygnDbqsJzsoJUg7KCcWzEtNV3qtoxcLnBkZiQ="))
$Py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$BuildScript = Join-Path $ProjectRoot "scripts\v3_build_corpus_ocr.py"
$IndexScript = Join-Path $ProjectRoot "scripts\v3_train_knowledge.py"

$Corpus = Join-Path $ProjectRoot "data\corpus_master.jsonl"
$OutDir = Join-Path $ProjectRoot "data\v3_index"

if (!(Test-Path $Py)) {
    throw "python venv not found: $Py"
}
if (!(Test-Path $BuildScript)) {
    throw "script not found: $BuildScript"
}
if (!(Test-Path $IndexScript)) {
    throw "script not found: $IndexScript"
}
if (!(Test-Path $InputRoot)) {
    throw "input root not found: $InputRoot"
}

& $Py -m pip install --upgrade pip
& $Py -m pip install --upgrade `
    pymupdf pypdf sentence-transformers rank-bm25 bitsandbytes `
    rapidocr-onnxruntime opencv-python-headless

# 1) OCR + corpus/chunk rebuild (resume-safe)
& $Py $BuildScript `
    --input-root $InputRoot `
    --project-root $ProjectRoot `
    --resume `
    --include-regex $IncludeRegex `
    --dpi 340 `
    --quality-threshold 0.90 `
    --min-chars-for-textlayer 180 `
    --chunk-size 1000 `
    --chunk-overlap 120

if (!(Test-Path $Corpus)) {
    throw "corpus build failed; missing: $Corpus"
}

# 2) Dense index build for QA retrieval
& $Py $IndexScript `
    --corpus-path $Corpus `
    --out-dir $OutDir `
    --embed-model "intfloat/multilingual-e5-large-instruct" `
    --device auto `
    --batch-size 32 `
    --min-pages-per-source 120 `
    --min-quality 0.82 `
    --min-chars 160 `
    --chunk-size 900 `
    --chunk-overlap 140 `
    --keep-methods "textlayer,ocr"

