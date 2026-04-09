$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path (Join-Path $projectRoot ".run") "processes.json"
$managerLog = Join-Path (Join-Path $projectRoot ".run") "manager.log"

function Write-ManagerLog {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $managerLog -Value "[$timestamp] $Message"
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
        return $false
    }

    & taskkill /PID $ProcessId /T /F | Out-Null
    return $true
}

if (-not (Test-Path $pidFile)) {
    Write-Host "No managed services found."
    exit 0
}

try {
    $state = Get-Content $pidFile -Raw | ConvertFrom-Json
}
catch {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Removed invalid runtime state file."
    exit 0
}

$sessionId = $state.sessionId
if ($sessionId) {
    Write-ManagerLog "Stopping services for session $sessionId."
}

foreach ($entry in $state.processes.PSObject.Properties) {
    $name = $entry.Name
    $processId = [int]$entry.Value.pid

    if (Stop-ManagedProcess -ProcessId $processId) {
        Write-ManagerLog "Stopped $name (PID $processId)."
        Write-Host "Stopped $name (PID $processId)."
    }
    else {
        Write-ManagerLog "$name was not running (PID $processId)."
        Write-Host "$name was not running (PID $processId)."
    }
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
