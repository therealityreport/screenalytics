# Codex Slack Integration

## Enable the integration
1. Install the Codex Slack app from the OpenAI dashboard.
2. In Slack, authorize the workspace where Screenalytics incidents are discussed.
3. Map the Codex workspace to this repo by providing the GitHub URL and the default playbook (`agents/playbooks/update-docs-on-change.yaml`) in the integration settings.

## `/export` command → `export_screen_time`
- Create a Slack slash command `/export` that sends the current channel thread as context to Codex.
- Configure the command to call the MCP tool `export_screen_time` exposed by the `screenalytics` MCP server.
- Require arguments: `episode_id`, `format` (csv/json), and optional `person_slug`.

## Example trigger
```
/export episode_id=S05E03 format=csv person_slug=meredith-marks
```
Codex relays the request to `export_screen_time`, uploads the generated CSV back to the thread, and posts a summary with the S3 link.
