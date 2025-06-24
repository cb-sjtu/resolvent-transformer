# Reference Code

This folder can store reference code from external projects. They can serve as helpful context for AI-assisted code generation and development.

## GitIngest

[GitIngest](https://gitingest.com/) can turn any Git repository into a simple text digest of its codebase. This is useful for feeding a codebase into any LLM.

The pre-commit hook `check-added-large-files` is disabled in this `reference` folder, so you can freely commit large digest files here.

It is strongly suggested to include an accompanying `README.md` file in this folder to explain the origin of the digest file. For example:

```
- file-name.txt
   - Source: name of external repository
   - Gitingest link: https://gitingest.com/...
   - Content: ...
```
