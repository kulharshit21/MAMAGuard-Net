param()

$ErrorActionPreference = "Stop"
$Remote = "gdrive:MAMAGuard-Net_Continuous_Backup_2026-11-01"
$Destination = "D:\MAMAGuard-Net\DGX_MIRROR"
$LogDirectory = Join-Path $Destination "_sync_logs"
$StatusPath = Join-Path $LogDirectory "latest_status.txt"
$MutexName = "Local\MAMAGuardDriveMirror"
$Timestamp = Get-Date -Format "yyyyMMddTHHmmss"
$LogPath = Join-Path $LogDirectory "drive_to_local_$Timestamp.log"
$CombinedPath = Join-Path $LogDirectory "drive_to_local_verify_$Timestamp.txt"

New-Item -ItemType Directory -Force -Path $Destination, $LogDirectory | Out-Null

$mutex = [System.Threading.Mutex]::new($false, $MutexName)
$acquired = $false

try {
    try {
        $acquired = $mutex.WaitOne(0)
    }
    catch [System.Threading.AbandonedMutexException] {
        $acquired = $true
    }

    if (-not $acquired) {
        $duplicateLog = Join-Path $LogDirectory "duplicate_launches.log"
        Add-Content -LiteralPath $duplicateLog `
            -Value "$((Get-Date).ToString('o')) sync already running; duplicate launch ignored" `
            -Encoding utf8
        exit 0
    }

    $rclone = (Get-Command rclone -ErrorAction Stop).Source

    @(
        "state=running"
        "started_at=$((Get-Date).ToString('o'))"
        "remote=$Remote"
        "destination=$Destination"
        "log=$LogPath"
        "verification=$CombinedPath"
    ) | Set-Content -LiteralPath $StatusPath -Encoding utf8

    & $rclone lsd $Remote --max-depth 1 *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Google Drive authentication or remote discovery failed with exit code $LASTEXITCODE"
    }

    & $rclone copy $Remote $Destination `
        --checksum `
        --create-empty-src-dirs `
        --transfers 4 `
        --checkers 16 `
        --fast-list `
        --retries 10 `
        --low-level-retries 20 `
        --retries-sleep 15s `
        --stats 60s `
        --stats-one-line-date `
        --log-level INFO `
        --log-file $LogPath `
        --exclude "**/*.tmp" `
        --exclude "**/*.part" `
        --exclude "**/*.partial" `
        --exclude "**/.nfs*" `
        --exclude "**/__pycache__/**" `
        --exclude "**/.pytest_cache/**" `
        --exclude "**/.ruff_cache/**" `
        --exclude "**/*.pyc"

    $copyExit = $LASTEXITCODE
    if ($copyExit -ne 0) {
        throw "Drive-to-local copy failed with exit code $copyExit"
    }

    & $rclone check $Remote $Destination `
        --size-only `
        --one-way `
        --fast-list `
        --combined $CombinedPath `
        --exclude "**/*.tmp" `
        --exclude "**/*.part" `
        --exclude "**/*.partial" `
        --exclude "**/.nfs*" `
        --exclude "**/__pycache__/**" `
        --exclude "**/.pytest_cache/**" `
        --exclude "**/.ruff_cache/**" `
        --exclude "**/*.pyc" `
        --log-level INFO `
        --log-file $LogPath

    $checkExit = $LASTEXITCODE
    if ($checkExit -ne 0) {
        throw "Post-copy verification found differences; rclone exit code $checkExit"
    }

    @(
        "state=success"
        "completed_at=$((Get-Date).ToString('o'))"
        "remote=$Remote"
        "destination=$Destination"
        "log=$LogPath"
        "verification=$CombinedPath"
        "copy_exit=0"
        "check_exit=0"
        "mode=one_way_copy_no_local_deletions"
    ) | Set-Content -LiteralPath $StatusPath -Encoding utf8
}
catch {
    $message = $_.Exception.Message
    Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) ERROR $message" -Encoding utf8
    @(
        "state=failed"
        "failed_at=$((Get-Date).ToString('o'))"
        "remote=$Remote"
        "destination=$Destination"
        "log=$LogPath"
        "verification=$CombinedPath"
        "error=$message"
    ) | Set-Content -LiteralPath $StatusPath -Encoding utf8
    exit 1
}
finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}

