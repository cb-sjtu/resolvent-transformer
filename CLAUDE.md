# Claude Code Instructions

## Commit Message Format

When creating commits, always include a detailed summary of the conversation in the commit message using this format:

```
<brief description of changes>

User requests:
- <user prompt/request (clarified) 1>
- <user prompt/request (clarified) 2>
- ...

User raw prompts:
- <exact user prompt 1>
- <exact user prompt 2>
- ...

Claude actions:
- <summary of what Claude did>
- <key files created/modified>
- <any analysis or decisions made>

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Guidelines

1. **User requests**: Only include user commands/requests that are relevant to THIS specific commit
2. **User raw prompts**: Include the exact user prompt/request that was used to generate the commit
3. **Claude actions**: Summarize the key actions taken, files modified, and decisions made for this commit
4. **Be specific**: Include file names, tool usage, and important findings
5. **Keep it concise**: Focus on the most important aspects of the conversation
5. **Check timing**: Compare git history (including timestamps) with Claude conversation history to determine which user requests belong to this commit

This helps maintain a clear history of how changes were made and what was discussed during the development process.

## Pre-commit Hooks

This repository uses pre-commit hooks that check your code when making commits. Understanding the workflow:

1. **Initial commit attempt**: When you run `git commit`, pre-commit hooks will automatically check your code
2. **Hook failures**: If code doesn't pass checks, the commit will be rejected
3. **Auto-fix attempts**: Pre-commit hooks will try to automatically fix issues (e.g., formatting, trailing whitespace)
4. **Re-stage and retry**: If hooks modify files, you must:
   - Run `git add <modified-files>` to stage the auto-fixed changes
   - Run `git commit` again with the same message
5. **Manual fixes**: If auto-fix doesn't work, manually adjust code according to the error messages and retry

### Common Pre-commit Hook Actions
- `trim trailing whitespace` - Removes extra spaces at line ends
- `fix end of files` - Ensures files end with newline
- `ruff` - Python code formatting and linting
- `check yaml/toml` - Validates configuration file syntax

### Best Practice
Always check `git status` after a failed commit to see which files were modified by hooks, then re-stage and commit again.

## Git Push Policy

**NEVER auto-push commits to remote repositories.** Always let the user decide when to push changes.

- Only create local commits when requested
- Never run `git push` unless explicitly asked by the user
- After successful commits, inform the user they can push when ready
- Respect the user's workflow and timing for sharing changes
