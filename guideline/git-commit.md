# Git Commit Instructions

## Commit Messages

When creating commits, always include a detailed summary of the conversation in the commit message using this format:

```
<brief description of changes>

Changes made:
- (AI): <concise summary of what Claude did>
- (User): <manual edits or user-authored code, if any>
- (AI): <additional Claude action if substantially different>

User raw prompts:
- <exact user prompt 1>
- <exact user prompt 2>

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Guidelines for Commit Messages

1. **Changes made**: Keep it concise, try to reduce the bullet points to the most important ones.
2. **User raw prompts**: Include the exact user prompt/request that was used to generate the commit. Of course, ignore the user's requests that are not relevant to this commit. Also ignore those too long, e.g., error messages, code blocks, etc.
3. In the end, don't add something like "Generated with [Claude Code](https://claude.ai/code)". Coauthor is enough.

### Detecting User vs Claude Changes

When composing commit messages, compare the changes to be committed (`git diff`) against Claude's actions in the current conversation. Any changes not made by Claude's recorded tool usage are user changes.

## Pre-commit Hooks

This repository uses pre-commit hooks that check your code when making commits. Claude should handle the workflow automatically:

1. **Initial commit attempt**: When you run `git commit`, pre-commit hooks will automatically check your code
2. **Hook failures**: If code doesn't pass checks, the commit will be rejected
3. **Auto-fix attempts**: Pre-commit hooks will try to automatically fix issues (e.g., formatting, trailing whitespace)
4. **Auto re-stage and retry**: If hooks modify files, Claude should automatically:
   - Run `git add <modified-files>` to stage the auto-fixed changes
   - Run `git commit` again with the same message
   - **DO NOT ask the user for permission** - handle this automatically
5. **Manual fixes**: If auto-fix doesn't work, manually adjust code according to the error messages and retry automatically

### Claude Behavior
- Always handle pre-commit hook failures automatically without user intervention
- Check `git status` after failed commits to see which files were modified by hooks
- Automatically re-stage and retry commits when hooks make changes
- Only notify the user if manual code changes are required that cannot be automated

## Git Push Policy

**NEVER auto-push commits to remote repositories.** Always let the user decide when to push changes.

- Only create local commits when requested
- Never run `git push` unless explicitly asked by the user
- After successful commits, inform the user they can push when ready
- Respect the user's workflow and timing for sharing changes
