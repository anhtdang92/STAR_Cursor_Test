# Auto-push script for git
$ErrorActionPreference = "Stop"

# Function to check if there are changes
function HasChanges {
    $status = git status --porcelain
    return $status -ne $null
}

# Function to commit and push changes
function CommitAndPush {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git add .
    git commit -m "Auto-commit at $timestamp"
    git push
}

# Main loop
while ($true) {
    try {
        if (HasChanges) {
            Write-Host "Changes detected at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            CommitAndPush
            Write-Host "Changes pushed successfully"
        }
        Start-Sleep -Seconds 300  # Check every 5 minutes
    }
    catch {
        Write-Host "Error occurred: $_"
        Start-Sleep -Seconds 60  # Wait a minute before retrying
    }
} 