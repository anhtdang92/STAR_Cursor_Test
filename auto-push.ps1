# Auto Git Push Script
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commitMessage = "Auto commit at $timestamp"

# Add all changes
git add .

# Commit changes
git commit -m "$commitMessage"

# Push to remote repository
git push

Write-Host "Changes have been automatically committed and pushed to the remote repository." 