# rt_log.ps1 -- run a command on the TRAINING PC (Windows PowerShell, inside the
# conda env) and capture its full output into rt_logs/<NAME>.txt. At the end a
# SHORT version (rt_logs/<NAME>.short.txt, via shorten_rt_log.py: boot noise
# and report repeats out, the run's --out JSON in) is placed on the clipboard.
#
# Usage:
#   .\scripts\rt_log.ps1 RT-1 python scripts/author_workcell.py
#   .\scripts\rt_log.ps1 RT-1 python scripts/zero_agent.py --task Ur5e-Insertion-Direct-v0 --num_envs 4 --headless
#
# The way back to the laptop is MANUAL and deliberately so:
#   1. run the command here, the log lands on the clipboard
#   2. on the laptop, paste it into rt_logs/inbox.txt (always the same file)
#   3. /rt-check -- a subagent filters and judges it, the main chat never
#      reads the log text
#
# This wrapper used to commit and push the log. That half was removed on
# 2026-08-22: git cost four round trips (PROBLEMS.md, 2026-08-21) and never
# addressed the actual goal, which is keeping the log out of the chat context.
# Logs are not versioned any more; rt_logs/VERDICTS.md is the record.
#
# Same NAME appends to the same file, so one RT's commands share one log.
#
# Two things this script is careful about, each one a bug that already cost a
# round trip:
#
#   * Python buffers its stdout when it is not a console, so script prints can
#     vanish while Isaac's own C++ output still shows. `python` gets `-u`
#     inserted automatically.
#   * The log file is written UTF-8 without BOM through an explicit writer.
#     Tee-Object would write UTF-16 in PowerShell 5.1.

# Deliberately a PLAIN param block with no [Parameter()] attributes. Attributes
# turn this into an advanced script, and an advanced script tries to BIND every
# dash-argument of the wrapped command to a parameter of its own -- so
# `rt_log.ps1 RT-9 python -c "print(1)"` dies with "no positional parameter
# accepts -c". Without the attributes the leftovers land in $args untouched.
param($Name)
$Command = @($args)

$ErrorActionPreference = 'Continue'

if (-not $Name -or $Command.Count -eq 0) {
    Write-Host "usage: .\scripts\rt_log.ps1 <NAME> <command...>"
    Write-Host "  e.g. .\scripts\rt_log.ps1 RT-5 python scripts/zero_agent.py --num_envs 4 --headless"
    Write-Host "  self-test: .\scripts\rt_log.ps1 PROBE cmd /c echo hallo"
    exit 2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoRoot 'rt_logs'
$logFile = Join-Path $logDir "$Name.txt"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$exe = $Command[0]
if ($Command.Count -gt 1) { $rest = $Command[1..($Command.Count - 1)] } else { $rest = @() }

# Unbuffered python, so print() lines cannot be swallowed on an abort.
if ($exe -match '^python[0-9.]*(\.exe)?$' -and ($rest -notcontains '-u')) {
    $rest = @('-u') + $rest
}

$head = git -C $repoRoot rev-parse --short HEAD 2>$null
if (-not $head) { $head = 'unknown' }

# ---------------------------------------------------------------- run the job
$writer = New-Object System.IO.StreamWriter($logFile, $true, (New-Object System.Text.UTF8Encoding($false)))
$code = 0
$runStart = Get-Date
# The run name reaches train.py as RT_NAME and lands in the run folder name.
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

# ------------------------------------------------- hand the log to the laptop
# THE SHORT LOG (2026-09-05): scripts/shorten_rt_log.py cuts the Isaac boot
# noise and the repeated startup reports out of the full log and appends the
# wrapped script's `--out` JSON when it is small. The SHORT file goes on the
# clipboard; the full log stays on disk as the fallback. The full log is
# rebuilt into the short one every time, so a NAME reused for several
# commands still gives one short file with every run in it.
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

# Set-Clipboard can fail in a session with no window station (a service, a
# remote non-interactive shell). That must not hide the log, so it is only a
# convenience -- the file path is printed either way.
$clip = 'Inhalt ist in der Zwischenablage.'
try { Get-Content $clipSource -Raw | Set-Clipboard -ErrorAction Stop }
catch { $clip = 'Zwischenablage ging nicht -- Datei von Hand oeffnen und kopieren.' }

# ONE FOLDER FOR THE RUN (user, 2026-09-13): the full and the short log are
# copied into the run folder this command created under logs/rsl_rl/<exp>/,
# next to the checkpoints, demo_metrics.json and scalars.csv (train.py).
# Guard on EQUALITY: exactly one folder newer than the start, else nothing
# is copied and the reason is printed.
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
Write-Host "[rt_log] Am Laptop in rt_logs\inbox.txt einfuegen, dann /rt-check"

exit $code
