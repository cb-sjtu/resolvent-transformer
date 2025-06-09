#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

#!/bin/bash
# run by `sh git-sync.sh` or `sh git-sync.sh --hard`
# This script will sync the local and origin/main branch.
# Use --hard to force sync (reset --hard) instead of merge.

# Parse command line arguments
HARD_SYNC=false
for arg in "$@"
do
    case $arg in
        --hard)
        HARD_SYNC=true
        shift
        ;;
    esac
done

if [ "$HARD_SYNC" = true ]; then
    echo "This script will force local and origin/main to mirror upstream/main..."
    echo "----------------------------------------------------------------------------------"
else
    echo "This script will sync the local and origin/main branch with upstream/main..."
    echo "Add --hard to force mirror instead of merge."
    echo "----------------------------------------------------------------------------------"
fi

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

# pull the current branch
git pull origin $current_branch --no-edit

# Check if upstream remote exists
if git remote | grep -q upstream; then
    echo "----------------------------------------------------------------------------------"
    if [ "$HARD_SYNC" = true ]; then
        echo "Detected upstream repository, forcing local and origin/main to mirror upstream/main..."
        git fetch upstream main
        git checkout main
        git reset --hard upstream/main
        git push -f origin main
    else
        echo "Detected upstream repository, syncing upstream/main to local and origin/main..."
        git fetch upstream main
        git fetch origin main
        git checkout main
        git merge origin/main --no-edit
        git merge upstream/main --no-edit
        git push origin main
    fi
    echo "----------------------------------------------------------------------------------"
else
    echo "----------------------------------------------------------------------------------"
    if [ "$HARD_SYNC" = true ]; then
        echo "No upstream repository detected, forcing local/main to mirror origin/main..."
        git fetch origin main
        git checkout main
        git reset --hard origin/main
    else
        echo "No upstream repository detected, syncing from origin/main..."
        git fetch origin main
        git checkout main
        git merge origin/main --no-edit
    fi
    echo "If this repository is forked from scaling-core, consider adding an upstream remote with"
    echo "git remote add upstream https://github.com/scaling-group/scaling-core.git"
    echo "----------------------------------------------------------------------------------"
fi

# Switch back to original branch
echo "Switching back to original branch $current_branch..."
git checkout "$current_branch"

# Merge changes from main branch
echo "Merging changes from main branch..."
git merge main --no-edit

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
echo "Current branch: $current_branch, you can push the changes to remote yourself."
