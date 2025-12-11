# Claude Code GitHub Action

This repository uses the [Claude Code GitHub Action](https://github.com/anthropics/claude-code-action) to enable AI-assisted development directly in GitHub Issues and Pull Requests.

## What It Does

When you tag `@claude` in a comment on an Issue or Pull Request, a GitHub Actions workflow runs Claude Code against the repository. Claude can:

- Review code and provide feedback
- Answer questions about the codebase
- Implement changes based on your request
- Fix bugs or failing tests
- Explain code behavior

Claude responds directly in the same thread where you tagged it.

## One-Time Setup Checklist

Before the action will work, a repository admin must complete these steps:

1. **Install the Claude GitHub App**
   - Go to: https://github.com/apps/claude
   - Install it on the `therealityreport/screenalytics` repository
   - Required permissions: Contents (RW), Issues (RW), Pull Requests (RW)

2. **Add the Anthropic API Key as a Secret**
   - Go to: Repository Settings > Secrets and variables > Actions
   - Add a new repository secret:
     - Name: `ANTHROPIC_API_KEY`
     - Value: Your Anthropic API key from [console.anthropic.com](https://console.anthropic.com)

## How to Use

### Basic Usage

Tag `@claude` anywhere in your comment to trigger the action:

```
@claude What does the `cluster_identities` function do?
```

### Code Review

Ask Claude to review a PR:

```
@claude /review

Please focus on:
- Error handling
- Performance implications
- API compatibility
```

### Implementation Requests

Ask Claude to implement changes:

```
@claude implement a rate limiter for the API endpoints in apps/api/routers/episodes.py
```

### Bug Fixes

Ask Claude to fix issues:

```
@claude fix the failing test in tests/ml/test_cluster.py - the assertion on line 45 is incorrect
```

### Explaining Code

Ask Claude to explain complex code:

```
@claude explain how the ByteTrack integration works in py_screenalytics/tracking/
```

## Where It Works

The action triggers when `@claude` appears in:

| Location | Trigger |
|----------|---------|
| Issue comment | Comment body contains `@claude` |
| PR conversation comment | Comment body contains `@claude` |
| PR review comment (on diff) | Comment body contains `@claude` |
| New issue | Issue body or title contains `@claude` |

## Cost Control

The workflow is designed to minimize unnecessary API usage:

- **Only triggers on `@claude`**: Comments without the tag do not start a run
- **No automatic runs**: Claude only responds when explicitly tagged
- **Concise prompts recommended**: Shorter, focused asks produce faster and cheaper responses

### Tips to Reduce Costs

1. Be specific about what you want (e.g., "review the error handling in `detect.py`" vs "review everything")
2. Break large requests into smaller, focused tasks
3. Provide context when relevant (file paths, line numbers, error messages)

## Troubleshooting

### Action Doesn't Run

- Verify the Claude GitHub App is installed on this repo
- Check that `ANTHROPIC_API_KEY` secret is set
- Confirm your comment contains `@claude` (exact match, case-sensitive)

### Action Runs But Claude Doesn't Respond

- Check the Actions tab for workflow run errors
- Verify the API key is valid and has sufficient credits
- Review the workflow logs for specific error messages

### Claude's Response Is Cut Off

- Very long responses may be truncated by GitHub's comment limits
- Break your request into smaller parts for complete responses

## Workflow File

The workflow is defined in [.github/workflows/claude-code.yml](../../.github/workflows/claude-code.yml).
