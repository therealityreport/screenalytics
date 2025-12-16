# Claude Code Integration Guide

This document explains how to use Claude Code effectively with Screenalytics.

## Quick Start

Claude Code automatically picks up context from:

1. **Root CLAUDE.md** - Project overview, tech stack, conventions
2. **Module CLAUDE.md files** - Detailed guidance for specific areas
3. **Skills** - Specialized workflows for common tasks
4. **Slash Commands** - Quick actions

## CLAUDE.md Files

| File | Purpose |
|------|---------|
| `/CLAUDE.md` | Project overview, key directories, coding conventions |
| `/apps/api/CLAUDE.md` | API architecture, services, manifest formats |
| `/apps/workspace-ui/CLAUDE.md` | Streamlit patterns, state management, UX rules |

### How They Work

When you open this repo in Claude Code, it automatically reads CLAUDE.md files to understand:
- Project structure and tech stack
- Coding conventions and patterns
- Common tasks and where to find things

## Available Skills

Skills are specialized guides for common workflows. Use them by asking Claude to use a specific skill.

### Pipeline Debug
```
Use the pipeline-debug skill to diagnose why episode rhobh-s05e02 is stuck.
```

Covers:
- Failed/stuck episode diagnosis
- Manifest validation
- Log analysis
- Threshold verification

### Faces Review UX
```
Use the faces-review-ux skill. I need to add a new metric badge.
```

Covers:
- UI component patterns
- Similarity badge colors
- Confirmation dialogs
- Cache invalidation

### Cluster Quality
```
Use the cluster-quality skill to analyze the clustering results.
```

Covers:
- Metrics interpretation
- Quality thresholds
- Diagnostic steps
- Common fixes

### Storage Health
```
Use the storage-health skill. Thumbnails aren't loading.
```

Covers:
- S3/MinIO diagnostics
- Presigned URL issues
- Path resolution
- Disk space

## Slash Commands

Quick actions for common tasks:

### /episode-status
```
/episode-status rhobh-s05e02
```

Checks pipeline completion status for an episode.

### /test
```
/test ml      # ML tests
/test api     # API tests
/test all     # All tests
/test syntax  # Syntax check only
```

Runs tests for specific areas.

## Best Practices

### When Starting Work

1. Tell Claude what you're working on
2. Reference the relevant skill if applicable
3. Ask Claude to check CLAUDE.md for conventions

Example:
```
I'm fixing a bug in the Smart Suggestions page. Use the faces-review-ux skill.
The similarity badges aren't showing the right colors.
```

### For Pipeline Issues

1. Use `/episode-status` first
2. Then use `pipeline-debug` skill if issues found
3. Reference specific log markers (`[PHASE]`, `[JOB]`, `[GUARDRAIL]`)

### For UI Changes

1. Use `faces-review-ux` skill
2. Follow the patterns in `apps/workspace-ui/CLAUDE.md`
3. Always use unique widget keys
4. Add confirmation dialogs for destructive actions

### For Quality Issues

1. Use `cluster-quality` skill
2. Check metrics against thresholds
3. Review `track_metrics.json`

## Configuration

Settings are in `.claude/settings.json`:

```json
{
  "model": "claude-sonnet-4-20250514",
  "contextPaths": [
    "CLAUDE.md",
    "AGENTS.md",
    "apps/api/CLAUDE.md",
    "apps/workspace-ui/CLAUDE.md"
  ],
  "skills": {
    "enabled": true,
    "directory": ".claude/skills"
  },
  "commands": {
    "enabled": true,
    "directory": ".claude/commands"
  }
}
```

## Directory Structure

```
.claude/
├── settings.json          # Project settings
├── settings.local.json    # Local overrides (gitignored)
├── commands/
│   ├── episode-status.md  # /episode-status command
│   └── test.md            # /test command
└── skills/
    ├── pipeline-debug/
    │   └── SKILL.md
    ├── faces-review-ux/
    │   └── SKILL.md
    ├── cluster-quality/
    │   └── SKILL.md
    └── storage-health/
        └── SKILL.md
```

## Updating Documentation

### To Update CLAUDE.md

Edit the relevant file and test by asking Claude about the topic you documented.

### To Add a New Skill

1. Create directory: `.claude/skills/{skill-name}/`
2. Add `SKILL.md` with:
   - When to use
   - Diagnostic steps
   - Key files
   - Checklist
3. Reference it in conversations: "Use the {skill-name} skill"

### To Add a Slash Command

1. Create `.claude/commands/{command}.md`
2. Document usage, arguments, examples
3. Use with `/{command} [args]`

## Troubleshooting

### Claude doesn't pick up CLAUDE.md

- Ensure file is in repo root or module directory
- Check `.claude/settings.json` has correct `contextPaths`
- Restart Claude Code session

### Skill not recognized

- Verify SKILL.md exists in `.claude/skills/{name}/`
- Check `skills.enabled: true` in settings
- Reference skill explicitly in prompt

### Command not working

- Verify .md file exists in `.claude/commands/`
- Check `commands.enabled: true` in settings
- Use exact command name from filename

## Related Documentation

- [CLAUDE.md](/CLAUDE.md) - Project overview
- [AGENTS.md](/AGENTS.md) - Agent behavior rules
- [apps/api/CLAUDE.md](/apps/api/CLAUDE.md) - API module guide
- [apps/workspace-ui/CLAUDE.md](/apps/workspace-ui/CLAUDE.md) - UI module guide
