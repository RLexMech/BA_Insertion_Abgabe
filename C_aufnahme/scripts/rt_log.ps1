param($Name)
$Command = @($args)

$ErrorActionPreference = 'Continue'

if (-not $Name -or $Command.Count -eq 0) {
    Write-Host "usage: .\scripts\rt_log.ps1 <NAME> <command...>"
    Write-Host "  e.g. .\scripts\rt_log.ps1 RT-902 python scripts/rsl_rl/play.py --help"
    Write-Host "  self-test: .\scripts\rt_log.ps1 PROBE cmd /c echo hallo"
    exit 2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoRoot 'rt_logs'
$logFile = Join-Path $logDir "$Name.txt"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$exe = $Command[0]
if ($Command.Count -gt 1) { $rest = $Command[1..($Command.Count - 1)] } else { $rest = @() }


if ($exe -match '^python[0-9.]*(\.exe)?$' -and ($rest -notcontains '-u')) {
    $rest = @('-u') + $rest
}

$head = git -C $repoRoot rev-parse --short HEAD 2>$null
if (-not $head) { $head = 'unknown' }


$writer = New-Object System.IO.StreamWriter($logFile, $true, (New-Object System.Text.UTF8Encoding($false)))
$code = 0
$runStart = Get-Date

$env:RT_NAME = $Name
try {
    foreach ($line in @(
        '==============================================================',
        "[rt_log] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  cwd: $(Get-Location)",
        "[rt_log] cmd: $exe $($rest -join ' ')",
        "[rt_log] git: $head",
        '==============================================================')) {
        Write-Host $line
        $writer.WriteLine($line)
    }
    $writer.Flush()

    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        $miss = "[rt_log] '$exe' not found on PATH. Is the conda env active? (bash/WSL has no conda env.)"
        Write-Host $miss
        $writer.WriteLine($miss)
        $code = 127
    } else {
        & $exe @rest 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            $writer.WriteLine($line)
            $writer.Flush()
        }
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    }

    $tail = "[rt_log] exit code: $code"
    Write-Host $tail
    $writer.WriteLine($tail)
}
finally {
    $writer.Close()
}


$shortFile = Join-Path $logDir "$Name.short.txt"
$shortArgs = @($logFile, '--out', $shortFile)
$outIdx = [array]::IndexOf($rest, '--out')
if ($outIdx -ge 0 -and ($outIdx + 1) -lt $rest.Count) {
    $shortArgs += @('--json', $rest[$outIdx + 1])
}
$shortener = Join-Path $PSScriptRoot 'shorten_rt_log.py'
$clipSource = $logFile
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $shortener @shortArgs
    if ($LASTEXITCODE -eq 0 -and (Test-Path $shortFile)) { $clipSource = $shortFile }
    else { Write-Host "[rt_log] shorten_rt_log.py failed (exit $LASTEXITCODE) -- full log goes to the clipboard" }
}


$clip = 'Inhalt ist in der Zwischenablage.'
try { Get-Content $clipSource -Raw | Set-Clipboard -ErrorAction Stop }
catch { $clip = 'Zwischenablage ging nicht -- Datei von Hand oeffnen und kopieren.' }


$runRoot = Join-Path $repoRoot 'logs/rsl_rl'
if (Test-Path $runRoot) {
    $newDirs = @(Get-ChildItem -Path $runRoot -Directory -Recurse -Depth 1 | Where-Object { $_.CreationTime -ge $runStart -and $_.Parent.FullName -ne $runRoot })
    if ($newDirs.Count -eq 1) {
        Copy-Item $logFile -Destination $newDirs[0].FullName -Force
        if (Test-Path $shortFile) { Copy-Item $shortFile -Destination $newDirs[0].FullName -Force }
        Write-Host "[rt_log] Log kopiert nach: $($newDirs[0].FullName)"
    } else {
        Write-Host "[rt_log] Log NICHT in einen Run-Ordner kopiert: $($newDirs.Count) neue Ordner seit Start (erwartet genau 1)"
    }
}

Write-Host "[rt_log] Log (voll): $logFile"
Write-Host "[rt_log] Log (kurz): $clipSource"
Write-Host "[rt_log] $clip"
Write-Host "[rt_log] Pruefung: Abschlussmeldung und Exitcode im vollstaendigen Log kontrollieren."

exit $code
