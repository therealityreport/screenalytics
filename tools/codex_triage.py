#!/usr/bin/env python3
import json
import pathlib
import re
import textwrap
from collections import defaultdict

RC = pathlib.Path("pr4.review_comments.json")
IC = pathlib.Path("pr4.issue_comments.json")


def load_jsonl_or_json(path: pathlib.Path):
    data = []
    try:
        text = path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed = [parsed]
            data = parsed
        except json.JSONDecodeError:
            arrays = re.findall(r"\[\s*{.*?}\s*]", text, re.S)
            for chunk in arrays:
                data += json.loads(chunk)
    except FileNotFoundError:
        pass
    return data


review = load_jsonl_or_json(RC)
issue_comments = load_jsonl_or_json(IC)

SUG_RE = re.compile(r"```suggestion(?P<lang>[^\n]*)\n(?P<body>.*?)```", re.S | re.M)


def severity_of(text: str) -> str:
    match = re.search(r"\b(P0|P1|P2)\b", text)
    return match.group(1) if match else "P1"


def category_of(text: str) -> str:
    lowered = text.lower()
    if any(
        k in lowered
        for k in ["secret", "token", "api key", "credential", "pii", "sanitize"]
    ):
        return "security"
    if any(
        k in lowered
        for k in [
            "none",
            "null",
            "exception",
            "try:",
            "validate",
            "bounds",
            "off-by-one",
            "race",
            "deadlock",
        ]
    ):
        return "correctness"
    if any(
        k in lowered
        for k in [
            "perf",
            "performance",
            "cache",
            "batch",
            "vectoriz",
            "o(",
            "complexity",
        ]
    ):
        return "performance"
    if any(
        k in lowered
        for k in [
            "typo",
            "style",
            "lint",
            "format",
            "black",
            "flake8",
            "eslint",
            "prettier",
        ]
    ):
        return "style"
    return "other"


def violates_invariants(text: str, suggestion: str):
    lowered = (text + "\n" + suggestion).lower()
    if "retinaface" in lowered and any(
        bad in lowered for bad in ["yolo", "coco", "ssd"]
    ):
        return "Keep RetinaFace for face detection per project guideline"
    if "arcface" in lowered and any(
        k in lowered for k in ["facenet", "dlib", "openface"]
    ):
        return "Keep ArcFace embeddings (512-d) for compatibility"
    if "bytetrack" in lowered and "strongsort" in lowered and "replace" in lowered:
        return "ByteTrack is required baseline; avoid wholesale replacement"
    if "s3" in lowered and any(
        k in lowered for k in ["change layout", "rename bucket", "flatten structure"]
    ):
        return "Respect S3 v2 layout contract"
    if "facebank" in lowered and any(
        k in lowered for k in ["non-atomic", "partial write"]
    ):
        return "Facebank updates must stay atomic"
    return None


def verdict_of(body: str, suggestion: str):
    severity = severity_of(body)
    category = category_of(body)
    invariant = violates_invariants(body, suggestion)
    if invariant:
        return "Skip", f"Conflicts with project invariant: {invariant}"
    if category in {"security", "correctness"}:
        return "Apply", f"{category} fix ({severity})"
    if category == "style":
        return "Apply", "low-risk formatting/nits"
    if category == "performance":
        suggestion_lower = suggestion.lower()
        if re.search(r"\b(np\\.|\btensor|\bvectoriz|\bbatch)", suggestion_lower):
            return "Consider", "perf change—verify benchmarks or add test"
        return "Consider", "perf tweak; confirm trade-offs"
    return "Consider", "needs human judgement"


items = []
for comment in review:
    path = comment.get("path", "(unknown)")
    body = comment.get("body", "")
    for match in SUG_RE.finditer(body):
        suggestion_body = match.group("body").strip()
        verdict, reason = verdict_of(body, suggestion_body)
        snippet_lines = suggestion_body.splitlines()
        snippet = "\n".join(snippet_lines[:12])
        if len(snippet_lines) > 12:
            snippet += "..."
        items.append(
            {
                "path": path,
                "verdict": verdict,
                "reason": reason,
                "snippet": snippet,
                "url": comment.get("html_url", ""),
            }
        )

for comment in review:
    if "```suggestion" in comment.get("body", ""):
        continue
    path = comment.get("path", "(unknown)")
    body = comment.get("body", "")
    verdict, reason = verdict_of(body, "")
    snippet = textwrap.shorten(body, width=180, placeholder="…")
    items.append(
        {
            "path": path,
            "verdict": verdict,
            "reason": reason,
            "snippet": snippet,
            "url": comment.get("html_url", ""),
        }
    )

by_file = defaultdict(list)
for item in items:
    by_file[item["path"]].append(item)

output_lines = ["# Codex PR #4 — Suggested changes triage\n"]
for path, arr in sorted(by_file.items()):
    output_lines.append(f"\n## {path}\n")
    for entry in arr:
        output_lines.append(f"- [{entry['verdict']}] {entry['reason']}\n")
        output_lines.append(f"  - {entry['url']}\n")
        output_lines.append("  - ```\n")
        output_lines.append(f"{entry['snippet']}\n")
        output_lines.append("  ```\n")

pathlib.Path("codex_pr4_checklist.md").write_text(
    "\n".join(output_lines), encoding="utf-8"
)
print("Wrote codex_pr4_checklist.md with triage results.")
