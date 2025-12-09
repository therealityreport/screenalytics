Check the pipeline status for an episode.

## Usage

`/episode-status <ep_id>`

Example: `/episode-status rhobh-s05e02`

## What This Does

1. Reads the episode manifest directory at `data/manifests/{ep_id}/`
2. Checks each pipeline stage for completion:
   - **Detect**: `faces.jsonl` exists
   - **Track**: `tracks.jsonl` exists
   - **Embed**: embeddings present
   - **Cluster**: `identities.json` exists
   - **Cast**: `cast_links.json` exists
3. Reports any missing artifacts or errors
4. Shows quality metrics summary if available

## Expected Output

```
Episode: rhobh-s05e02
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pipeline Status:
  ✅ Detect    - 1,234 faces
  ✅ Track     - 456 tracks
  ✅ Embed     - embeddings ready
  ✅ Cluster   - 78 clusters (23 singletons)
  ⚠️  Cast     - 45/78 linked

Quality Metrics:
  Identity Similarity: 0.82 (good)
  Singleton Fraction: 29.5% (ok)
  Cast Coverage: 57.7%
```

## Arguments

- `ep_id` (required): Episode identifier in format `{show}-s{season}e{episode}`

## Related Commands

- `/test api` - Run API tests
- Use `pipeline-debug` skill for deeper diagnosis
