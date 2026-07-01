param(
    [string]$Prompt = "Explain in 5 bullet points why caching helps API latency.",
    [int]$RunsPerConfig = 3,
    [int]$SessionTurns = 1,
    [int]$PromptRepeat = 1,
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 8005
)

$ErrorActionPreference = "Stop"
$root = (Join-Path $PSScriptRoot "..")
Set-Location $root

$pythonExe = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Python venv not found at .venv\\Scripts\\python.exe. Run scripts/start-windows.ps1 first."
    exit 1
}

if (-not (Test-Path "models/gemma-4-E2B-it.litertlm")) {
    Write-Warning "Local model path models/gemma-4-E2B-it.litertlm was not found. Set MODEL_PATH before running this benchmark if needed."
}

function Wait-Health {
    param(
        [string]$HealthUrl,
        [int]$TimeoutSeconds = 180
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $HealthUrl -Method GET -TimeoutSec 3
            if ($resp.StatusCode -eq 200) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Test-HealthUp {
    param(
        [string]$HealthUrl
    )

    try {
        $resp = Invoke-WebRequest -Uri $HealthUrl -Method GET -TimeoutSec 2
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Resolve-BenchmarkPort {
    param(
        [string]$TargetHost,
        [int]$PreferredPort,
        [int]$MaxAttempts = 20
    )

    $candidate = $PreferredPort
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $healthUrl = "http://$TargetHost`:$candidate/healthz"
        if (-not (Test-HealthUp -HealthUrl $healthUrl)) {
            if ($candidate -ne $PreferredPort) {
                Write-Host "Port $PreferredPort is busy. Using free port $candidate for benchmark."
            }
            return $candidate
        }
        $candidate++
    }

    throw "Could not find a free benchmark port starting at $PreferredPort."
}

$benchmarkPort = Resolve-BenchmarkPort -TargetHost $ServerHost -PreferredPort $Port
$baseUrl = "http://$ServerHost`:$benchmarkPort"
$healthUrl = "$baseUrl/healthz"
$modelsUrl = "$baseUrl/v1/models"
$chatUrl = "$baseUrl/v1/chat/completions"

$effectivePrompt = ((($Prompt + " ") * [Math]::Max(1, $PromptRepeat))).Trim()

$configs = @(
    @{ Name = "default"; Value = $null },
    @{ Name = "2048"; Value = "2048" },
    @{ Name = "8192"; Value = "8192" },
    @{ Name = "16384"; Value = "16384" }
)

$results = @()

foreach ($cfg in $configs) {
    Write-Host "`n=== Running config: $($cfg.Name) ==="

    $oldContext = $env:LITERT_MAX_NUM_TOKENS
    $oldModelPath = $env:MODEL_PATH
    $oldServerPort = $env:SERVER_PORT

    if ($cfg.Value) {
        $env:LITERT_MAX_NUM_TOKENS = $cfg.Value
    } else {
        Remove-Item Env:LITERT_MAX_NUM_TOKENS -ErrorAction SilentlyContinue
    }

    $effectiveContext = if ($env:LITERT_MAX_NUM_TOKENS) { $env:LITERT_MAX_NUM_TOKENS } else { "SDK default" }
    Write-Host "Context setting for this config: $effectiveContext"

    if (-not $env:MODEL_PATH) {
        $env:MODEL_PATH = "models/gemma-4-E2B-it.litertlm"
    }
    $env:SERVER_PORT = "$benchmarkPort"

    $logBase = Join-Path $env:TEMP ("litert-bench-{0}-{1}" -f $cfg.Name, [guid]::NewGuid().ToString("N"))
    $stdoutLog = "$logBase.out.log"
    $stderrLog = "$logBase.err.log"

    $server = Start-Process -FilePath $pythonExe `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", $ServerHost, "--port", "$benchmarkPort" `
        -WorkingDirectory $root `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    Start-Sleep -Milliseconds 800
    if ($server.HasExited) {
        $stderrTail = ""
        if (Test-Path $stderrLog) {
            $stderrTail = (Get-Content $stderrLog -Tail 20) -join "`n"
        }
        $stdoutTail = ""
        if (Test-Path $stdoutLog) {
            $stdoutTail = (Get-Content $stdoutLog -Tail 20) -join "`n"
        }
        throw "Failed to start benchmark server for config $($cfg.Name).`nSTDERR:`n$stderrTail`nSTDOUT:`n$stdoutTail"
    }

    try {
        if (-not (Wait-Health -HealthUrl $healthUrl)) {
            throw "Server did not become healthy for config $($cfg.Name)."
        }

        $modelResp = Invoke-RestMethod -Uri $modelsUrl -Method GET -TimeoutSec 30
        $modelId = $modelResp.data[0].id
        if (-not $modelId) {
            throw "Could not resolve model id from /v1/models"
        }

        $runRows = @()
        for ($i = 1; $i -le $RunsPerConfig; $i++) {
            $messages = @()

            for ($turn = 1; $turn -le [Math]::Max(1, $SessionTurns); $turn++) {
                $messages += @{ role = "user"; content = "$effectivePrompt [turn $turn]" }

                $payload = @{
                    model = $modelId
                    stream = $false
                    temperature = 0
                    messages = $messages
                } | ConvertTo-Json -Depth 12

                $sw = [System.Diagnostics.Stopwatch]::StartNew()
                $resp = Invoke-RestMethod -Uri $chatUrl -Method POST -ContentType "application/json" -Body $payload -TimeoutSec 300
                $sw.Stop()

                $elapsed = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
                $completionTokens = [int]$resp.usage.completion_tokens
                $promptTokens = [int]$resp.usage.prompt_tokens
                $tps = [Math]::Round($completionTokens / $elapsed, 2)
                $assistantText = [string]$resp.choices[0].message.content
                $messages += @{ role = "assistant"; content = $assistantText }

                $row = [PSCustomObject]@{
                    config = $cfg.Name
                    run = $i
                    turn = $turn
                    elapsed_s = [Math]::Round($elapsed, 2)
                    prompt_tokens = $promptTokens
                    completion_tokens = $completionTokens
                    tokens_per_s = $tps
                }
                $runRows += $row
                $results += $row

                Write-Host (("Run {0} Turn {1}: {2}s, prompt={3}, completion={4}, tps={5}" -f $i, $turn, $row.elapsed_s, $promptTokens, $completionTokens, $tps))
            }
        }

        $avgElapsed = [Math]::Round((($runRows | Measure-Object -Property elapsed_s -Average).Average), 2)
        $avgTps = [Math]::Round((($runRows | Measure-Object -Property tokens_per_s -Average).Average), 2)
        Write-Host (("AVG [{0}] -> elapsed={1}s, tps={2}" -f $cfg.Name, $avgElapsed, $avgTps))
    }
    finally {
        if ($server -and -not $server.HasExited) {
            Stop-Process -Id $server.Id -Force
            Start-Sleep -Milliseconds 800
        }

        if (Test-Path $stdoutLog) {
            Remove-Item $stdoutLog -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrLog) {
            Remove-Item $stderrLog -ErrorAction SilentlyContinue
        }

        if ([string]::IsNullOrEmpty($oldContext)) {
            Remove-Item Env:LITERT_MAX_NUM_TOKENS -ErrorAction SilentlyContinue
        } else {
            $env:LITERT_MAX_NUM_TOKENS = $oldContext
        }

        if ([string]::IsNullOrEmpty($oldModelPath)) {
            Remove-Item Env:MODEL_PATH -ErrorAction SilentlyContinue
        } else {
            $env:MODEL_PATH = $oldModelPath
        }

        if ([string]::IsNullOrEmpty($oldServerPort)) {
            Remove-Item Env:SERVER_PORT -ErrorAction SilentlyContinue
        } else {
            $env:SERVER_PORT = $oldServerPort
        }
    }
}

Write-Host "`n=== Final comparative table ==="
$results | Sort-Object config, run | Format-Table -AutoSize

Write-Host "`n=== Average by config ==="
$results |
    Group-Object config |
    ForEach-Object {
        $avgElapsed = [Math]::Round((($_.Group | Measure-Object -Property elapsed_s -Average).Average), 2)
        $avgTps = [Math]::Round((($_.Group | Measure-Object -Property tokens_per_s -Average).Average), 2)
        [PSCustomObject]@{
            config = $_.Name
            avg_elapsed_s = $avgElapsed
            avg_tokens_per_s = $avgTps
        }
    } |
    Sort-Object config |
    Format-Table -AutoSize
