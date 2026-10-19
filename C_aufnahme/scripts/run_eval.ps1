param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('T-final','T-final-F','T-nominal','T-nominal-F')]
    [string]$Block,
    [Parameter(Mandatory = $true)][ValidatePattern('^RT-[0-9]+$')][string]$Rt,
    [string]$RunRoot = '',
    [string]$ResultsRoot = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunRoot) { $RunRoot = Join-Path $repoRoot 'models' }
if (-not $ResultsRoot) { $ResultsRoot = Join-Path $repoRoot 'results/eval' }
Push-Location $repoRoot
try {
    # Resolve relative arguments against C, not the native process working directory.
    if (-not [IO.Path]::IsPathRooted($RunRoot)) { $RunRoot = Join-Path $repoRoot $RunRoot }
    if (-not [IO.Path]::IsPathRooted($ResultsRoot)) { $ResultsRoot = Join-Path $repoRoot $ResultsRoot }
    $RunRoot = [IO.Path]::GetFullPath($RunRoot)
    $ResultsRoot = [IO.Path]::GetFullPath($ResultsRoot)
    $manifest = Get-Content -LiteralPath (Join-Path $repoRoot 'models/manifest.json') -Raw | ConvertFrom-Json
    $policies = @($manifest.models | Where-Object { $_.method -ne 'fixeddr' })
    if ($Block -like '*-F') { $policies = @($manifest.models | Where-Object { $_.method -eq 'fixeddr' }) }
    $table = 'scripts/eval/eval_table_seed10.csv'
    $width = 'target'
    if ($Block -like 'T-nominal*') { $width = 'nominal' }
    $tableHash = (Get-FileHash -LiteralPath $table -Algorithm SHA256).Hash.ToLower()
    $evalName = "eval_${width}_$($tableHash.Substring(0,8))_$Rt"
    $head = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read Git HEAD.' }
    $plan = git log -1 --format=%H -- eval_plan.md
    if (-not $DryRun) {
        if (-not $plan) { throw 'Commit eval_plan.md before evaluating.' }
        if (@(git status --porcelain --untracked-files=no).Count) { throw 'Tracked changes: commit or resolve them before evaluating.' }
    }
    $jobs = @()
    foreach ($p in $policies) {
        $seed = [int]($p.run_id -replace '^.*s','')
        $name = "$($p.method)_s$seed"
        $checkpoint = Join-Path (Join-Path $RunRoot $p.folder) $p.checkpoint
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw "Missing checkpoint: $checkpoint" }
        if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLower() -ne $p.sha256) { throw "Checkpoint differs from archived evaluation: $checkpoint. Run git lfs pull if this is an LFS pointer." }
        $evalDir = Join-Path (Split-Path -Parent $checkpoint) $evalName
        $destination = Join-Path (Join-Path $ResultsRoot $Block) $name
        $logName = "${Rt}_$name"
        $log = Join-Path $repoRoot "rt_logs/$logName.txt"
        foreach ($path in @($evalDir,$destination,$log)) {
            if (Test-Path -LiteralPath $path) { throw "Output already exists: $path. Use a new -Rt and, for a repeated block, a new -ResultsRoot." }
        }
        $command = @('python','-u','scripts/rsl_rl/play.py','--task','Ur5e-Insertion-Direct-v0','--headless',
            '--seed','100','--checkpoint',$checkpoint,'--eval-table',$table,'--eval-width',$width,'--eval-tag',$Rt,
            'env.start_floor_m=0.020')
        $jobs += @{ Name=$logName; Command=$command; EvalDir=$evalDir; Destination=$destination; Log=$log }
    }
    Write-Host "[run_eval] $Block; $($jobs.Count) policies; $table; width=$width; seed=100; HEAD=$head"
    foreach ($job in $jobs) {
        if ($DryRun) {
            Write-Host "[run_eval] DRY RUN $($job.Name): $($job.Command -join ' ')"
            Write-Host "[run_eval] -> $($job.Destination)"
            continue
        }
        if ((git rev-parse HEAD) -ne $head -or @(git status --porcelain --untracked-files=no).Count) { throw 'Git state changed during evaluation.' }
        if ((Get-FileHash -LiteralPath $table -Algorithm SHA256).Hash.ToLower() -ne $tableHash) { throw 'Evaluation table changed.' }
        $command = $job.Command
        & (Join-Path $PSScriptRoot 'rt_log.ps1') $job.Name @command
        $code = $LASTEXITCODE
        if ($null -eq $code -or $code -ne 0) { throw "Run $($job.Name) failed (exit $code). See $($job.Log)" }
        foreach ($file in @('episodes.csv','episodes_meta.json','applied.csv','applied_meta.json','eval_manifest.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $job.EvalDir $file))) { throw "Missing result $file in $($job.EvalDir)" }
        }
        Copy-Item -LiteralPath $job.Log -Destination $job.EvalDir
        $shortLog = $job.Log -replace '\.txt$','.short.txt'
        if (Test-Path -LiteralPath $shortLog) { Copy-Item -LiteralPath $shortLog -Destination $job.EvalDir }
        [IO.File]::WriteAllText((Join-Path $job.EvalDir 'table_filehash.txt'), "$tableHash`n")
        [IO.File]::WriteAllText((Join-Path $job.EvalDir 'head.txt'), "$head`n")
        New-Item -ItemType Directory -Path (Split-Path -Parent $job.Destination) -Force | Out-Null
        Copy-Item -LiteralPath $job.EvalDir -Destination $job.Destination -Recurse
    }
    Write-Host "[run_eval] $Block complete: $($jobs.Count) runs; DryRun=$DryRun"
} catch {
    Write-Host "[run_eval] STOP: $_"
    exit 1
} finally {
    Pop-Location
}
exit 0
