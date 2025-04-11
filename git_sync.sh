#!/bin/bash
# run by `sh git_sync.sh`

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
    echo "Detected upstream repository, syncing upstream/main to origin/main..."
    git fetch upstream main
    git checkout main
    git merge upstream/main
    git push origin main
    echo "----------------------------------------------------------------------------------"
else
    echo "----------------------------------------------------------------------------------"
    echo "No upstream repository detected, skipping upstream/main sync"
    echo "If this repository is forked from scaling-core, consider adding an upstream remote with"
    echo "git remote add upstream https://github.com/scaling-group/scaling-core.git"
    echo "----------------------------------------------------------------------------------"
fi

# Pull latest code from origin/main
echo "Pulling latest code from origin/main..."
git checkout main
git pull origin main

# Switch back to original branch
echo "Switching back to original branch $current_branch..."
git checkout "$current_branch"

# Merge changes from main branch
echo "Merging changes from main branch..."
git merge main

# If there was a stash, attempt to pop it
if [ -z "$stash_skipped" ]; then
    echo "----------------------------------------------------------------------------------"
    echo "Attempting to restore stash..."
    if ! git stash pop; then
        echo "  Warning: stash pop failed"
        echo "  - Manual conflict resolution may be needed"
    fi
    echo "----------------------------------------------------------------------------------"
else
    echo "No previous stash, skipping pop operation"
fi

echo "All operations completed!"