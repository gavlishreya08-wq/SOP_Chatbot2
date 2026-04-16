$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeDir = Join-Path $projectRoot ".run"
$pidFile = Join-Path $runtimeDir "processes.json"
$managerLog = Join-Path $runtimeDir "manager.log"

function Write-ManagerLog {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $managerLog -Value "[$timestamp] $Message"
}

function Get-PythonExecutable {
    $candidates = @(
        (Join-Path $projectRoot "venv\Scripts\python.exe"),
        (Join-Path $projectRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    throw "Python executable was not found."
}

function Get-NpmExecutable {
    foreach ($name in @("npm.cmd", "npm")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw "npm was not found on PATH. Install Node.js before starting the app."
}

function Get-LanAddresses {
    try {
        $configs = Get-NetIPConfiguration | Where-Object {
            $_.NetAdapter.Status -eq "Up" -and $_.IPv4Address -and $_.IPv4DefaultGateway
        }
    }
    catch {
        return @()
    }

    $addresses = foreach ($config in $configs) {
        foreach ($address in $config.IPv4Address) {
            $ip = $address.IPAddress
            if ($ip -and $ip -notlike "127.*" -and $ip -notlike "169.254.*") {
                $ip
            }
        }
    }

    return @($addresses | Sort-Object -Unique)
}

function Test-ManagedProcessRunning {
    param([int]$ProcessId)

    try {
        $null = Get-Process -Id $ProcessId -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Stop-ManagedProcess {
    param([int]$ProcessId)

    if (-not (Test-ManagedProcessRunning -ProcessId $ProcessId)) {
        return
    }

    & taskkill /PID $ProcessId /T /F | Out-Null
}

if (Test-Path $pidFile) {
    try {
        $existingState = Get-Content $pidFile -Raw | ConvertFrom-Json
    }
    catch {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        $existingState = $null
    }

    if ($existingState) {
        $activeProcesses = @()
        foreach ($entry in $existingState.processes.PSObject.Properties) {
            $processId = [int]$entry.Value.pid
            if (Test-ManagedProcessRunning -ProcessId $processId) {
                $activeProcesses += "$($entry.Name) (PID $processId)"
            }
        }

        if ($activeProcesses.Count -gt 0) {
            throw "Managed services are already running: $($activeProcesses -join ', '). Run .\stop.ps1 first."
        }
    }
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$backendLogDir = Join-Path $runtimeDir "backend"
$frontendLogDir = Join-Path $runtimeDir "frontend"
New-Item -ItemType Directory -Path $backendLogDir -Force | Out-Null
New-Item -ItemType Directory -Path $frontendLogDir -Force | Out-Null

$sessionId = Get-Date -Format "yyyyMMdd-HHmmss"
$backendOutLog = Join-Path $backendLogDir "$sessionId.out.log"
$backendErrLog = Join-Path $backendLogDir "$sessionId.err.log"
$frontendOutLog = Join-Path $frontendLogDir "$sessionId.out.log"
$frontendErrLog = Join-Path $frontendLogDir "$sessionId.err.log"
$pythonExe = Get-PythonExecutable
$npmExe = Get-NpmExecutable
$startedPids = @()

Write-ManagerLog "Starting services for session $sessionId."
Write-ManagerLog "Backend logs: $backendOutLog | $backendErrLog"
Write-ManagerLog "Frontend logs: $frontendOutLog | $frontendErrLog"

try {
    $backendProcess = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $backendOutLog `
        -RedirectStandardError $backendErrLog `
        -PassThru
    $startedPids += $backendProcess.Id

    $frontendProcess = Start-Process `
        -FilePath $npmExe `
        -ArgumentList @("run", "dev", "--", "--host", "0.0.0.0", "--port", "5173") `
        -WorkingDirectory (Join-Path $projectRoot "frontend") `
        -RedirectStandardOutput $frontendOutLog `
        -RedirectStandardError $frontendErrLog `
        -PassThru
    $startedPids += $frontendProcess.Id

    Start-Sleep -Seconds 2

    foreach ($process in @($backendProcess, $frontendProcess)) {
        if ($process.HasExited) {
            throw "A managed service exited during startup. Check the logs in $runtimeDir."
        }
    }
}
catch {
    foreach ($processId in $startedPids) {
        Stop-ManagedProcess -ProcessId $processId
    }
    Write-ManagerLog "Startup failed for session $sessionId. See service logs for details."
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    throw
}

$state = [ordered]@{
    sessionId = $sessionId
    startedAt = (Get-Date).ToString("o")
    managerLog = $managerLog
    processes = [ordered]@{
        backend = [ordered]@{
            pid = $backendProcess.Id
            cwd = $projectRoot
            stdoutLog = $backendOutLog
            stderrLog = $backendErrLog
            command = @($pythonExe, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload")
        }
        frontend = [ordered]@{
            pid = $frontendProcess.Id
            cwd = (Join-Path $projectRoot "frontend")
            stdoutLog = $frontendOutLog
            stderrLog = $frontendErrLog
            command = @($npmExe, "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173")
        }
    }
}

$state | ConvertTo-Json -Depth 6 | Set-Content $pidFile -Encoding UTF8

Write-ManagerLog "Backend started with PID $($backendProcess.Id)."
Write-ManagerLog "Frontend started with PID $($frontendProcess.Id)."

$lanAddresses = Get-LanAddresses

Write-Host "Started development servers:"
Write-Host "  backend  (local): PID $($backendProcess.Id) -> http://127.0.0.1:8000"
Write-Host "  frontend (local): PID $($frontendProcess.Id) -> http://127.0.0.1:5173"
if ($lanAddresses.Count -gt 0) {
    Write-Host "Wi-Fi/LAN URLs:"
    foreach ($address in $lanAddresses) {
        Write-Host "  backend  : http://${address}:8000"
        Write-Host "  frontend : http://${address}:5173"
    }
}
else {
    Write-Host "No active LAN IPv4 address detected. Connect to Wi-Fi or Ethernet to expose the app on your local network."
}
Write-Host "Session: $sessionId"
Write-Host "Logs: $backendOutLog, $backendErrLog, $frontendOutLog, $frontendErrLog"
Write-Host "Manager log: $managerLog"
Write-Host "Use .\stop.ps1 to stop both services."
