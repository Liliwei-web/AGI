# 归档脚本：在官方环境里跑一局 train（本地评分可见），用于验证解法。
# 需要主办方的仿真客户端与赛题系统；本仓库不包含它们，此脚本仅为流程存档。
#
# 用法：
#   $env:ARENA_RELEASE = "C:\path\to\arena\release"   # 含 start_test.ps1 的目录
#   $env:AGENT_DIR     = "C:\path\to\baseline-agent-master"
#   $env:UV_EXE        = "uv"
#   .\run_train_solve.ps1

param(
    [string]$Task = 'competition-preliminary-jigsaw-task',
    [string]$Mode = 'train'
)

$ErrorActionPreference = 'Continue'
$release = $env:ARENA_RELEASE
$agentDir = $env:AGENT_DIR
$uv = if ($env:UV_EXE) { $env:UV_EXE } else { 'uv' }
$logs = if ($env:RUNLOG_DIR) { $env:RUNLOG_DIR } else { Join-Path $agentDir 'logs' }

if (-not $release -or -not (Test-Path (Join-Path $release 'start_test.ps1'))) {
    Write-Output 'ARENA_RELEASE-NOT-FOUND'; exit 3
}
if (-not $agentDir -or -not (Test-Path $agentDir)) {
    Write-Output 'AGENT_DIR-NOT-FOUND'; exit 3
}

$stamp = Get-Date -Format 'HHmmss'
$srvOut = Join-Path $logs ("solve_srv_out_$stamp.log")
$srvErr = Join-Path $logs ("solve_srv_err_$stamp.log")
$agentLog = Join-Path $logs ("solve_agent_$stamp.log")
$evalPath = Join-Path (Join-Path $release 'arena_offline') 'eval_res.json'

Get-Process arena_offline, tongsim_server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 1) 起服务器，等到它开始出题
$server = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $release 'start_test.ps1'), $Mode, '--task', $Task) `
    -WorkingDirectory $release -WindowStyle Hidden -RedirectStandardOutput $srvOut -RedirectStandardError $srvErr -PassThru
Write-Output ('SERVER-PID=' + $server.Id)

$ready = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $srvErr) {
        if ((Get-Content $srvErr -Raw -ErrorAction SilentlyContinue) -match 'Starting subject') { $ready = $true; break }
    }
}
if (-not $ready) { Write-Output 'SERVER-NOT-READY'; taskkill /PID $server.Id /T /F *> $null; exit 2 }
Write-Output 'SERVER-READY'

# 2) 起 agent（SOLVE=1 -> 只走解题路径）
$env:SOLVE = '1'
$t0 = if (Test-Path $evalPath) { (Get-Item $evalPath).LastWriteTime } else { [datetime]'2000-01-01' }
Set-Location $agentDir
& $uv run arenaagent --agent_name jigsaw_probe_agent --config config.toml --run_times 1 2>&1 |
    Out-File -FilePath $agentLog -Encoding UTF8
Write-Output ('AGENT-EXIT=' + $LASTEXITCODE)

# 3) 等本地评分落盘并打印
$deadline = (Get-Date).AddMinutes(8)
$ok = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if ((Test-Path $evalPath) -and ((Get-Item $evalPath).LastWriteTime -gt $t0)) { $ok = $true; break }
    if (-not (Get-Process -Id $server.Id -ErrorAction SilentlyContinue)) { $ok = $true; break }
}
Write-Output ('EVAL-READY=' + $ok)

Start-Sleep -Seconds 2
taskkill /PID $server.Id /T /F *> $null
if (Test-Path $evalPath) { Write-Output ('EVAL=' + ((Get-Content $evalPath -Raw) -replace '\s+', ' ')) }
Write-Output ('AGENTLOG=' + $agentLog)