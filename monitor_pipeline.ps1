$root = "C:\Users\LENOVO\Desktop\ML-DNA\dna-adaptive-redundancy\dna-adaptive-redundancy"
$logFile = "$root\logs\_monitor.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Add-Content -Path $logFile -Value $line
    Write-Host $line
}

Log "=== Monitor started ==="
Set-Location $root

while ($true) {
    $py = Get-Process python -ErrorAction SilentlyContinue
    if (-not $py) {
        Log "Pipeline not running - restarting..."
        Start-Process python -ArgumentList "run_pipeline.py --from-stage datasets --keep-going" -NoNewWindow -RedirectStandardOutput "logs\_top_level.log" -RedirectStandardError "logs\_top_level_err.log"
        Log "Restarted."
    } else {
        $status = python run_pipeline.py --status 2>&1 | Out-String
        if ($status -match "Pipeline complete") {
            Log "=== PIPELINE COMPLETE ==="
            break
        }
        $summary = ($status -split "`n" | Select-String "Summary:").Line
        $tail = (Get-Content logs\_top_level.log -Tail 2 -ErrorAction SilentlyContinue) -join " | "
        Log "OK. $summary | $tail"
    }
    Start-Sleep -Seconds 300
}
