# run_eval.ps1 -- run ONE evaluation block (10 play.py runs, eval_plan.md
# section 3.2) on the TRAINING PC, each run through rt_log.ps1.
#
# Usage (repo root, conda env active):
#   .\scripts\run_eval.ps1 -Block T-nominal -Rt RT-222
#   .\scripts\run_eval.ps1 -Block T-nominal -Rt RT-222 -DryRun   # checks only, no run
#
# Pull ONCE before the block, never during it (eval_collect.py refuses a block
# whose runs disagree on code_commit). The script holds the clone to that:
#
#   before the block  HEAD readable, no tracked change (the SAME rule as the
#                     manifest's code_dirty: git status --porcelain
#                     --untracked-files=no, tools/eval_manifest.py:120),
#                     the table is there, all 10 checkpoints are there, no
#                     eval folder and no rt_log file of this block exists yet.
#                     Every problem is listed, then exit 1.
#   before each run   HEAD unchanged and still no tracked change, else exit 1.
#   after each run    the run's full and short log, the table's Get-FileHash
#                     (table_filehash.txt) and HEAD (head.txt) go into the
#                     run's eval folder. rt_log exit != 0 -> exit with it.
#
# Table and width per block are READ from eval_collect.py BLOCKS (their one
# home), not typed here. T-grenze (one policy), technik (practice table) and
# the Fixed-DR blocks T-nominal-F / T-final-F (five other policies, D-192) are
# refused: they are not ten-policy blocks.
#
# Run names: <Rt>_<autodr|nodr>_s<N>, e.g. RT-222_autodr_s1. The laptop copies
# each eval folder to results\eval\<Block>\<autodr|nodr>_s<N>\ (eval_collect.py
# _FOLDER). -RunRoot exists for the laptop dry-run test only.

param(
    [Parameter(Mandatory = $true)][string]$Block,
    [Parameter(Mandatory = $true)][string]$Rt,
    [string]$RunRoot = '',
    [switch]$DryRun
)

# Continue, as in rt_log.ps1: under 'Stop' PowerShell 5.1 turns any native
# stderr line (git, python) into a terminating error. Every failure below is
# caught by an explicit equality check instead.
$ErrorActionPreference = 'Continue'

# eval_plan.md section 3.2 "Abspiel-Seed (Nutzer 2026-09-22)": every policy,
# every block, --seed 100.
$PlaySeed = 100

# The ten frozen policies (eval_plan.md "Kontext": model_1499.pt, AutoDR
# RT-206s1..s5, No-DR RT-207s1..s5). Folder names = log_dir in
# rt_logs/RT-206s<N>/RT-206s<N>_env.yaml and rt_logs/RT-207s<N>/RT-207s<N>_env.yaml.
# EXCEPTION RT-206s2..s4: env.yaml says "RT-206s1" in the folder name (the runs
# were started under that rt_log name), but on the training PC the folders are
# named RT-206s2/s3/s4 (user, 2026-09-22, after run_eval STOP "no checkpoint").
$Policies = @(
    @{ Method = 'autodr'; Seed = 1; Folder = '09-15_13-50-36__RT-206s1_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed1' },
    @{ Method = 'autodr'; Seed = 2; Folder = '09-15_18-05-48__RT-206s2_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed2' },
    @{ Method = 'autodr'; Seed = 3; Folder = '09-16_02-12-05__RT-206s3_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed3' },
    @{ Method = 'autodr'; Seed = 4; Folder = '09-16_15-10-35__RT-206s4_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed4' },
    @{ Method = 'autodr'; Seed = 5; Folder = '09-17_00-05-01__RT-206s5_autodr_offset5mm_currOFF_start+20mm_floor-30mm_seed5' },
    @{ Method = 'nodr';   Seed = 1; Folder = '09-17_09-58-45__RT-207s1_offset5mm_currOFF_start+20mm_floor-30mm_seed1' },
    @{ Method = 'nodr';   Seed = 2; Folder = '09-17_16-03-29__RT-207s2_offset5mm_currOFF_start+20mm_floor-30mm_seed2' },
    @{ Method = 'nodr';   Seed = 3; Folder = '09-17_22-41-29__RT-207s3_offset5mm_currOFF_start+20mm_floor-30mm_seed3' },
    @{ Method = 'nodr';   Seed = 4; Folder = '09-18_12-38-12__RT-207s4_offset5mm_currOFF_start+20mm_floor-30mm_seed4' },
    @{ Method = 'nodr';   Seed = 5; Folder = '09-18_20-38-33__RT-207s5_offset5mm_currOFF_start+20mm_floor-30mm_seed5' }
)
$Checkpoint = 'model_1499.pt'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunRoot) { $RunRoot = Join-Path $repoRoot 'logs\rsl_rl\ur5e_insertion' }

function Stop-Block([string[]]$Lines, [int]$Code = 1) {
    foreach ($l in $Lines) { Write-Host "[run_eval] STOP: $l" }
    Pop-Location
    exit $Code
}

function Get-Head { (git -C $repoRoot rev-parse HEAD 2>$null) }
function Get-Tracked { @(git -C $repoRoot status --porcelain --untracked-files=no 2>$null) }

Push-Location $repoRoot

# ------------------------------------------------------------ block -> table
if ($Rt -notmatch '^RT-[0-9]+$') { Stop-Block @("-Rt '$Rt' is not RT-<number>") }
if ($Block -in @('T-grenze', 'technik')) { Stop-Block @("block '$Block' is not a ten-policy block") }
# D-192: the Fixed-DR blocks play ONLY the five Fixed-DR policies, on 8299485d
# (eval_plan.md section 3.2); this script's ten policies must never run there.
if ($Block -like '*-F') { Stop-Block @("block '$Block' is a Fixed-DR block, not a ten-policy block") }
$collect = Join-Path $PSScriptRoot 'eval_collect.py'
$pattern = '^\s*"' + [regex]::Escape($Block) + '":\s*\("([^"]+)",\s*"(target|nominal)"\),'
$hits = @(Select-String -Path $collect -Pattern $pattern)
if ($hits.Count -ne 1) { Stop-Block @("eval_collect.py BLOCKS has $($hits.Count) lines for '$Block' (want exactly 1)") }
$tableRel = 'scripts/eval/' + $hits[0].Matches[0].Groups[1].Value
$width = $hits[0].Matches[0].Groups[2].Value
$tableAbs = Join-Path $repoRoot $tableRel

# ------------------------------------------------------------ pre-block checks
$problems = @()
$head = Get-Head
if (-not $head) { $problems += 'git rev-parse HEAD gave nothing' }
$tracked = Get-Tracked
if ($tracked.Count -ne 0) { $problems += "tracked changes in the clone: $($tracked -join '; ')" }
$tableHash = ''
if (-not (Test-Path -LiteralPath $tableAbs -PathType Leaf)) { $problems += "no table at $tableAbs" }
else { $tableHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $tableAbs).Hash }
if ($tableHash) { $evalDirName = "eval_${width}_$($tableHash.Substring(0, 8).ToLower())" }
else { $evalDirName = '' }   # tools/eval_manifest.py eval_dirname, no tag

foreach ($p in $Policies) {
    $p.Name = "${Rt}_$($p.Method)_s$($p.Seed)"
    $p.RunDir = Join-Path $RunRoot $p.Folder
    $p.Ckpt = Join-Path $p.RunDir $Checkpoint
    $p.Log = Join-Path $repoRoot "rt_logs\$($p.Name).txt"
    $p.ShortLog = Join-Path $repoRoot "rt_logs\$($p.Name).short.txt"
    if (-not (Test-Path -LiteralPath $p.Ckpt -PathType Leaf)) { $problems += "no checkpoint: $($p.Ckpt)" }
    if ($evalDirName) {
        $p.EvalDir = Join-Path $p.RunDir $evalDirName
        if (Test-Path -LiteralPath $p.EvalDir) { $problems += "eval folder exists already: $($p.EvalDir)" }
    }
    # rt_log.ps1 APPENDS to an existing log, so an old one would mix two runs.
    if (Test-Path -LiteralPath $p.Log) { $problems += "rt_log file exists already: $($p.Log)" }
}
$names = @($Policies | ForEach-Object { $_.Name } | Sort-Object -Unique)
if ($names.Count -ne 10) { $problems += "run names are not 10 distinct: $($names -join ', ')" }
if ($problems.Count -ne 0) { Stop-Block $problems }

Write-Host "[run_eval] block $Block, table $tableRel, width $width, seed $PlaySeed"
Write-Host "[run_eval] HEAD $head"
Write-Host "[run_eval] table Get-FileHash $tableHash"
Write-Host "[run_eval] eval folder name $evalDirName"
Write-Host "[run_eval] pre-block checks: 10 checkpoints, 0 eval folders, 0 rt_log files, clean -- OK"

# ------------------------------------------------------------ the ten runs
$i = 0
foreach ($p in $Policies) {
    $i++
    $now = Get-Head
    if ($now -ne $head) { Stop-Block @("HEAD moved from $head to $now before run $i ($($p.Name))") }
    $tracked = Get-Tracked
    if ($tracked.Count -ne 0) { Stop-Block @("tracked changes before run $i ($($p.Name)): $($tracked -join '; ')") }

    $cmd = @('python', '-u', 'scripts/rsl_rl/play.py', '--task', 'Ur5e-Insertion-Direct-v0', '--headless',
             '--seed', "$PlaySeed", '--checkpoint', $p.Ckpt,
             '--eval-table', $tableRel, '--eval-width', $width)
    Write-Host "[run_eval] run $i/10: $($p.Name)"
    if ($DryRun) {
        Write-Host "[run_eval]   DRY RUN: .\scripts\rt_log.ps1 $($p.Name) $($cmd -join ' ')"
        Write-Host "[run_eval]   DRY RUN: eval folder $($p.EvalDir)"
        continue
    }

    & (Join-Path $PSScriptRoot 'rt_log.ps1') $p.Name @cmd
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 1 }

    if (Test-Path -LiteralPath $p.EvalDir -PathType Container) {
        Copy-Item -LiteralPath $p.Log -Destination $p.EvalDir -Force
        if (Test-Path -LiteralPath $p.ShortLog) { Copy-Item -LiteralPath $p.ShortLog -Destination $p.EvalDir -Force }
        $hashNow = (Get-FileHash -Algorithm SHA256 -LiteralPath $tableAbs).Hash
        [System.IO.File]::WriteAllText((Join-Path $p.EvalDir 'table_filehash.txt'), "$hashNow`n", (New-Object System.Text.UTF8Encoding($false)))
        [System.IO.File]::WriteAllText((Join-Path $p.EvalDir 'head.txt'), "$(Get-Head)`n", (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "[run_eval]   log, table_filehash.txt, head.txt -> $($p.EvalDir)"
    } else {
        Stop-Block @("run $i ($($p.Name)) exit $code and no eval folder at $($p.EvalDir)") ([Math]::Max($code, 1))
    }
    if ($code -ne 0) { Stop-Block @("run $i ($($p.Name)) exit $code") $code }
}

if ($DryRun) {
    Write-Host "[run_eval] DRY RUN done: checks passed, nothing ran."
    Pop-Location
    exit 0
}
Write-Host "[run_eval] block $Block done: 10 of 10 runs, exit 0 each. Eval folders:"
foreach ($p in $Policies) { Write-Host "  $($p.EvalDir)  ->  results\eval\$Block\$($p.Method)_s$($p.Seed)" }
Pop-Location
exit 0
