$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

# Optional .env support for engine context override without changing path defaults.
if (Test-Path ".env") {
    foreach ($line in Get-Content ".env") {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $parts = $trimmed.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")

        if ($key -eq "LITERT_MAX_NUM_TOKENS") {
            if ($value) {
                # .env should win for deterministic context configuration.
                $env:LITERT_MAX_NUM_TOKENS = $value
            } else {
                Remove-Item Env:LITERT_MAX_NUM_TOKENS -ErrorAction SilentlyContinue
            }
        }
    }
}

$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/gemma-4-E2B-it.litertlm" }
$ServerPort = if ($env:SERVER_PORT) { $env:SERVER_PORT } else { "8005" }

if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found: $ModelPath"
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $created = $false

    try {
        py -3.12 -m venv .venv
        $created = $true
    } catch {
        Write-Host "Python 3.12 launcher not available, trying python..."
    }

    if (-not $created) {
        python -m venv .venv
    }
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "LiteRT Session Server"
Write-Host "Local:   http://127.0.0.1:$ServerPort/v1"
Write-Host "LAN:     http://<YOUR_WINDOWS_IP>:$ServerPort/v1"
Write-Host "Health:  http://127.0.0.1:$ServerPort/healthz"
if ($env:LITERT_MAX_NUM_TOKENS) {
    Write-Host "Context: LITERT_MAX_NUM_TOKENS=$($env:LITERT_MAX_NUM_TOKENS)"
} else {
    Write-Host "Context: SDK default"
}
Write-Host ""

$env:MODEL_PATH = $ModelPath
$env:SERVER_PORT = $ServerPort
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port $ServerPort
