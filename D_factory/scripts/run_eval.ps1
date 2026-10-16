param(
    [Parameter(Mandatory = $true)][string]$Block,
    [Parameter(Mandatory = $true)][string]$Rt,
    [string]$RunRoot = '',
    [switch]$DryRun
)


$ErrorActionPreference = 'Continue'


$PlaySeed = 100


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


if ($Rt -notmatch '^RT-[0-9]+$') { Stop-Block @("-Rt '$Rt' is not RT-<number>") }
if ($Block -in @('T-grenze', 'technik')) { Stop-Block @("block '$Block' is not a ten-policy block") }


if ($Block -like '*-F') { Stop-Block @("block '$Block' is a Fixed-DR block, not a ten-policy block") }
$collect = Join-Path $PSScriptRoot 'eval_collect.py'
$pattern = '^\s*"' + [regex]::Escape($Block) + '":\s*\("([^"]+)",\s*"(target|nominal)"\),'
$hits = @(Select-String -Path $collect -Pattern $pattern)
if ($hits.Count -ne 1) { Stop-Block @("eval_collect.py BLOCKS has $($hits.Count) lines for '$Block' (want exactly 1)") }
$tableRel = 'scripts/eval/' + $hits[0].Matches[0].Groups[1].Value
$width = $hits[0].Matches[0].Groups[2].Value
$tableAbs = Join-Path $repoRoot $tableRel


$problems = @()
$head = Get-Head
if (-not $head) { $problems += 'git rev-parse HEAD gave nothing' }
$tracked = Get-Tracked
if ($tracked.Count -ne 0) { $problems += "tracked changes in the clone: $($tracked -join '; ')" }
$tableHash = ''
if (-not (Test-Path -LiteralPath $tableAbs -PathType Leaf)) { $problems += "no table at $tableAbs" }
else { $tableHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $tableAbs).Hash }
if ($tableHash) { $evalDirName = "eval_${width}_$($tableHash.Substring(0, 8).ToLower())" }
else { $evalDirName = '' }

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


$i = 0
foreach ($p in $Policies) {
    $i++
    $now = Get-Head
    if ($now -ne $head) { Stop-Block @("HEAD moved from $head to $now before run $i ($($p.Name))") }
    $tracked = Get-Tracked
    if ($tracked.Count -ne 0) { Stop-Block @("tracked changes before run $i ($($p.Name)): $($tracked -join '; ')") }

    $cmd = @('python', '-u', 'scripts/rsl_rl/play.py', '--task', 'Ur5e-Insertion-Direct-v0', '--headless',
             '--seed', "$PlaySeed", '--checkpoint', $p.Ckpt,
             '--eval-table', $tableRel, '--eval-width', $width, 'env.start_floor_m=0.020')
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
