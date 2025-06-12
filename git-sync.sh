#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

#!/bin/bash
# run by `sh git-sync.sh`

echo "----------------------------------------------------------------------------------"
echo "This script will merge upstream/main to local and origin/main."
echo "----------------------------------------------------------------------------------"

echo "Checking for upstream repository..."
if ! git remote | grep -q upstream; then
    echo "No upstream repository detected, auto adding upstream scaling-group/scaling-core..."
    git remote add upstream https://github.com/scaling-group/scaling-core.git
    echo "Upstream repository added successfully."
else
    echo "Upstream repository already exists."
    if [ "$(git remote get-url upstream)" != "https://github.com/scaling-group/scaling-core.git" ]; then
        echo "Upstream URL is incorrect!"
        echo "Your upstream : $(git remote get-url upstream)"
        echo "Expected: https://github.com/scaling-group/scaling-core.git"
        echo "This script may not work for you."
        echo "Exiting..."
        exit 1
    else
        echo "Upstream repository is correct: $(git remote get-url upstream)"
    fi
fi

echo "----------------------------------------------------------------------------------"

# Save current branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Check for tracked changes (excluding untracked files)
if [ -n "$(git diff --name-only)" ]; then
    echo "Found tracked changes, executing git stash..."
    git stash -q
    echo "Stash executed successfully"
else
    echo "No tracked changes, skipping git stash"
    stash_skipped=true
fi

echo "----------------------------------------------------------------------------------"

# Check if upstream remote exists
echo "Syncing upstream/main to local and origin/main..."
echo "Fetching upstream/main and origin/main..."
git fetch upstream main -q
git fetch origin main -q
echo "Checking out main branch..."
git checkout main -q
echo "Merging origin/main to local main..."
git merge origin/main --no-edit -q
echo "Merging upstream/main to local main..."
git merge upstream/main -m "Merge branch 'scaling-group:main' into main" -q
echo "Pushing local main to origin/main..."
git push origin main -q

echo "----------------------------------------------------------------------------------"

# Switch back to original branch
echo "Switching back to original branch $current_branch..."
git checkout "$current_branch" -q

# If there was a stash, attempt to pop it
if [ -z "$stash_skipped" ]; then
    echo "Attempting to restore stash..."
    if git stash pop -q; then
        echo "Stash restored successfully"
    else
        echo "  Warning: stash pop failed"
        echo "  - Manual conflict resolution may be needed"
    fi
else
    echo "No previous stash, skipping pop operation"
fi
echo "----------------------------------------------------------------------------------"

echo "All operations completed!"
echo "Current branch: $current_branch."
