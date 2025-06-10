#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

#!/bin/bash
# run by `sh git-sync.sh`

echo "This script will merge upstream/main to local and origin/main."
echo "----------------------------------------------------------------------------------"

# Save current branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Check for tracked changes (excluding untracked files)
if [ -n "$(git diff --name-only)" ]; then
    echo "Found tracked changes, executing git stash..."
    git stash
else
    echo "No tracked changes, skipping git stash"
    stash_skipped=true
fi

# Check if upstream remote exists
if git remote | grep -q upstream; then
    echo "----------------------------------------------------------------------------------"
    echo "Detected upstream repository, syncing upstream/main to local and origin/main..."
    git fetch upstream main
    git fetch origin main
    git checkout main
    git merge origin/main --no-edit
    git merge upstream/main --no-edit
    git push origin main
    echo "----------------------------------------------------------------------------------"
else
    echo "----------------------------------------------------------------------------------"
    echo "No upstream repository detected, syncing from origin/main..."
    git fetch origin main
    git checkout main
    git merge origin/main --no-edit
    echo "If this repository is forked from scaling-core, consider adding an upstream remote with"
    echo "git remote add upstream https://github.com/scaling-group/scaling-core.git"
    echo "----------------------------------------------------------------------------------"
fi

# Switch back to original branch
echo "Switching back to original branch $current_branch..."
git checkout "$current_branch"

# If there was a stash, attempt to pop it
if [ -z "$stash_skipped" ]; then
    echo "----------------------------------------------------------------------------------"
    echo "Attempting to restore stash..."
    if git stash pop -q; then
        echo "Stash restored successfully"
    else
        echo "  Warning: stash pop failed"
        echo "  - Manual conflict resolution may be needed"
    fi
    echo "----------------------------------------------------------------------------------"
else
    echo "No previous stash, skipping pop operation"
fi

echo "All operations completed!"
echo "Current branch: $current_branch."
