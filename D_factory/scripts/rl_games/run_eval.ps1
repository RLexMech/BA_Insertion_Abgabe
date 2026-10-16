param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('technik', 'T-auswahl', 'T-final', 'T-nominal', 'T-final-F', 'T-nominal-F', 'T-grenze', 'T-ood')]
    [string]$Block,
    [Parameter(Mandatory = $true)][string]$Rt,
    [string]$Final = '',
    [ValidateSet('', 'pn2', 'gx2', 'py2')][string]$Noise = '',
    [string]$RunRoot = '',
    [switch]$DryRun
)


$ErrorActionPreference = 'Continue'


$PlaySeed = 100


$ArmPrefix = @{ autodr = 'RT-242'; fixeddr = 'RT-243'; nodr = 'RT-244' }
$CkptPattern = '^last_ur5e_insertion_ep_100_rew_[-+0-9.eE]+\.pth$'


$YawBelief = 'env.pocket_obs_yaw_rad=0.0223402'
$BaseOverrides = @('env.obs_wrench_mode=force_prevact', 'env.start_floor_m=0.020')


$NoiseOverride = @{
    pn2 = 'env.obs_noise_pocket_pos_std_m=0.01'
    gx2 = 'env.grasp_obs_offset_x_m=0.006'
    py2 = 'env.pocket_obs_yaw_rad=0.0446804'
}

$scriptsDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptsDir
if (-not $RunRoot) { $RunRoot = Join-Path $repoRoot 'logs\rl_games\ur5e_insertion' }

function Stop-Block([string[]]$Lines, [int]$Code = 1) {
    foreach ($l in $Lines) { Write-Host "[run_eval] STOP: $l" }
    Pop-Location
    exit $Code
}

function Get-Head { (git -C $repoRoot rev-parse HEAD 2>$null) }
function Get-Tracked { @(git -C $repoRoot status --porcelain --untracked-files=no 2>$null) }

Push-Location $repoRoot


if ($Rt -notmatch '^RT-[0-9]+$') { Stop-Block @("-Rt '$Rt' is not RT-<number>") }
if (($Block -eq 'T-grenze') -ne [bool]$Final) { Stop-Block @("-Final is given with -Block T-grenze and only there") }
if ($Final -and $Final -notmatch '^(autodr|nodr|fixeddr)_s[1-5]$') { Stop-Block @("-Final '$Final' is not <autodr|nodr|fixeddr>_s<1-5>") }
if ($Noise -and $Block -ne 'T-final') { Stop-Block @("-Noise runs only on the T-final table (-Block T-final)") }


function New-Run([string]$Method, [int]$Seed, [string]$Width, [string]$Tag) {
    @{ Method = $Method; Seed = $Seed; Width = $Width; Tag = $Tag }
}
$runs = @()
switch ($Block) {
    'technik' {
        $runs += New-Run 'autodr' 1 'target' ''
        $runs += New-Run 'autodr' 1 'target' 'rep2'
        $runs += New-Run 'autodr' 1 'nominal' 'nom'
    }
    'T-grenze' {
        $runs += New-Run ($Final -replace '_s[1-5]$', '') ([int]($Final -replace '^.*_s', '')) '' ''
    }
    { $_ -in @('T-final-F', 'T-nominal-F') } {
        foreach ($s in 1..5) { $runs += New-Run 'fixeddr' $s '' '' }
    }
    'T-ood' {
        foreach ($s in 1..5) { $runs += New-Run 'autodr' $s '' '' }
    }
    default {
        if ($Noise) {
            foreach ($s in 1..5) { $runs += New-Run 'autodr' $s '' $Noise }
        } else {
            foreach ($m in @('autodr', 'nodr')) { foreach ($s in 1..5) { $runs += New-Run $m $s '' '' } }
        }
    }
}


$collect = Join-Path $scriptsDir 'eval_collect.py'
$pattern = '^\s*"' + [regex]::Escape($Block) + '":\s*\("([^"]+)",\s*(?:"(target|nominal)"|None)\),'
$hits = @(Select-String -Path $collect -Pattern $pattern)
if ($hits.Count -ne 1) { Stop-Block @("eval_collect.py BLOCKS has $($hits.Count) lines for '$Block' (want exactly 1)") }
$tableRel = 'scripts/eval/' + $hits[0].Matches[0].Groups[1].Value
$blockWidth = $hits[0].Matches[0].Groups[2].Value
if ($Block -ne 'technik' -and -not $blockWidth) { Stop-Block @("eval_collect.py BLOCKS gives '$Block' no width") }
foreach ($r in $runs) { if (-not $r.Width) { $r.Width = $blockWidth } }
$tableAbs = Join-Path $repoRoot $tableRel


$problems = @()
$head = Get-Head
if (-not $head) { $problems += 'git rev-parse HEAD gave nothing' }
$tracked = Get-Tracked
if ($tracked.Count -ne 0) { $problems += "tracked changes in the clone: $($tracked -join '; ')" }
$tableHash = ''
if (-not (Test-Path -LiteralPath $tableAbs -PathType Leaf)) { $problems += "no table at $tableAbs" }
else { $tableHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $tableAbs).Hash.ToLower() }

foreach ($r in $runs) {
    $r.Name = "${Rt}_$($r.Method)_s$($r.Seed)"
    if ($r.Tag) { $r.Name += "_$($r.Tag)" }
    $r.Log = Join-Path $repoRoot "rt_logs\$($r.Name).txt"
    $r.ShortLog = Join-Path $repoRoot "rt_logs\$($r.Name).short.txt"
    $filter = "*__$($ArmPrefix[$r.Method])s$($r.Seed)_*"
    $dirs = @(Get-ChildItem -LiteralPath $RunRoot -Directory -Filter $filter -ErrorAction SilentlyContinue)
    if ($dirs.Count -ne 1) { $problems += "$($dirs.Count) run folders match $RunRoot\$filter (want exactly 1)"; continue }
    $r.RunDir = $dirs[0].FullName
    $ck = @(Get-ChildItem -LiteralPath (Join-Path $r.RunDir 'nn') -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match $CkptPattern })
    if ($ck.Count -ne 1) { $problems += "$($ck.Count) checkpoints match $CkptPattern in $($r.RunDir)\nn (want exactly 1)"; continue }
    $r.Ckpt = $ck[0].FullName
    if ($tableHash) {

        $r.EvalDir = Join-Path $r.RunDir "eval_$($r.Width)_$($tableHash.Substring(0, 8))"
        if ($r.Tag) { $r.EvalDir += "_$($r.Tag)" }
        if (Test-Path -LiteralPath $r.EvalDir) { $problems += "eval folder exists already: $($r.EvalDir)" }
    }

    if (Test-Path -LiteralPath $r.Log) { $problems += "rt_log file exists already: $($r.Log)" }
}
$names = @($runs | ForEach-Object { $_.Name } | Sort-Object -Unique)
if ($names.Count -ne $runs.Count) { $problems += "run names are not $($runs.Count) distinct: $($names -join ', ')" }
if ($problems.Count -ne 0) { Stop-Block $problems }

Write-Host "[run_eval] block $Block, table $tableRel, seed $PlaySeed, $($runs.Count) run(s)$(if ($Noise) { ", noise $Noise" })"
Write-Host "[run_eval] HEAD $head"
Write-Host "[run_eval] table Get-FileHash $tableHash"
Write-Host "[run_eval] pre-block checks: $($runs.Count) checkpoints found once, 0 eval folders, 0 rt_log files, clean -- OK"


$i = 0
foreach ($r in $runs) {
    $i++
    $now = Get-Head
    if ($now -ne $head) { Stop-Block @("HEAD moved from $head to $now before run $i ($($r.Name))") }
    $tracked = Get-Tracked
    if ($tracked.Count -ne 0) { Stop-Block @("tracked changes before run $i ($($r.Name)): $($tracked -join '; ')") }

    $cmd = @('python', '-u', 'scripts/rl_games/play.py', '--task', 'Ur5e-Insertion-Direct-v0', '--headless',
             '--seed', "$PlaySeed", '--checkpoint', $r.Ckpt,
             '--eval-table', $tableRel, '--eval-width', $r.Width)
    if ($r.Tag) { $cmd += @('--eval-tag', $r.Tag) }
    if ($Block -eq 'T-ood') { $cmd += '--eval-ood' }
    $cmd += $BaseOverrides
    if ($r.Tag -eq 'py2') { $cmd += $NoiseOverride['py2'] } else { $cmd += $YawBelief }
    if ($r.Tag -in @('pn2', 'gx2')) { $cmd += $NoiseOverride[$r.Tag] }

    Write-Host "[run_eval] run $i/$($runs.Count): $($r.Name)"
    if ($DryRun) {
        Write-Host "[run_eval]   DRY RUN: .\scripts\rt_log.ps1 $($r.Name) $($cmd -join ' ')"
        Write-Host "[run_eval]   DRY RUN: eval folder $($r.EvalDir)"
        continue
    }

    & (Join-Path $scriptsDir 'rt_log.ps1') $r.Name @cmd
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 1 }

    if (Test-Path -LiteralPath $r.EvalDir -PathType Container) {
        Copy-Item -LiteralPath $r.Log -Destination $r.EvalDir -Force
        if (Test-Path -LiteralPath $r.ShortLog) { Copy-Item -LiteralPath $r.ShortLog -Destination $r.EvalDir -Force }
        $hashNow = (Get-FileHash -Algorithm SHA256 -LiteralPath $tableAbs).Hash
        [System.IO.File]::WriteAllText((Join-Path $r.EvalDir 'table_filehash.txt'), "$hashNow`n", (New-Object System.Text.UTF8Encoding($false)))
        [System.IO.File]::WriteAllText((Join-Path $r.EvalDir 'head.txt'), "$(Get-Head)`n", (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "[run_eval]   log, table_filehash.txt, head.txt -> $($r.EvalDir)"
    } else {
        Stop-Block @("run $i ($($r.Name)) exit $code and no eval folder at $($r.EvalDir)") ([Math]::Max($code, 1))
    }
    if ($code -ne 0) { Stop-Block @("run $i ($($r.Name)) exit $code") $code }
}

if ($DryRun) {
    Write-Host "[run_eval] DRY RUN done: checks passed, nothing ran."
    Pop-Location
    exit 0
}
Write-Host "[run_eval] block $Block done: $($runs.Count) of $($runs.Count) runs, exit 0 each. Eval folders:"
foreach ($r in $runs) {
    $leaf = "$($r.Method)_s$($r.Seed)"
    if ($r.Tag) { $leaf += "_$($r.Tag)" }
    Write-Host "  $($r.EvalDir)  ->  results\eval\$Block\$leaf"
}
Pop-Location
exit 0
