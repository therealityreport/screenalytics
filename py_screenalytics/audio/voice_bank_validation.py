"""
Voice Bank Validation Report Generator.

Generates explicit reports when voice bank matching fails, with
recommendations for fixing issues and improving voice assignment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceBankIssue:
    """A single issue found during voice bank validation."""
    severity: str  # "error", "warning", "info"
    code: str
    message: str
    recommendation: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class VoiceBankValidationReport:
    """Comprehensive voice bank validation report."""
    ep_id: str
    show_id: str
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Voice bank stats
    voice_bank_path: Optional[str] = None
    voice_bank_exists: bool = False
    labeled_voices_count: int = 0
    labeled_voice_names: List[str] = field(default_factory=list)

    # Episode stats
    episode_clusters_count: int = 0
    episode_speakers_found: int = 0
    episode_speech_duration_s: float = 0.0

    # Matching stats
    matched_voices_count: int = 0
    unmatched_clusters_count: int = 0
    match_rate: float = 0.0

    # Issues and recommendations
    issues: List[VoiceBankIssue] = field(default_factory=list)
    overall_status: str = "unknown"  # "success", "partial", "failed"

    def add_issue(self, issue: VoiceBankIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ep_id": self.ep_id,
            "show_id": self.show_id,
            "generated_at": self.generated_at,
            "voice_bank_path": self.voice_bank_path,
            "voice_bank_exists": self.voice_bank_exists,
            "labeled_voices_count": self.labeled_voices_count,
            "labeled_voice_names": self.labeled_voice_names,
            "episode_clusters_count": self.episode_clusters_count,
            "episode_speakers_found": self.episode_speakers_found,
            "episode_speech_duration_s": self.episode_speech_duration_s,
            "matched_voices_count": self.matched_voices_count,
            "unmatched_clusters_count": self.unmatched_clusters_count,
            "match_rate": self.match_rate,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "recommendation": i.recommendation,
                    "details": i.details,
                }
                for i in self.issues
            ],
            "overall_status": self.overall_status,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Voice Bank Validation Report",
            f"",
            f"**Episode:** {self.ep_id}",
            f"**Show:** {self.show_id}",
            f"**Generated:** {self.generated_at}",
            f"",
            f"## Overall Status: {self.overall_status.upper()}",
            f"",
            f"## Voice Bank Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Voice Bank Path | {self.voice_bank_path or 'N/A'} |",
            f"| Voice Bank Exists | {'Yes' if self.voice_bank_exists else 'No'} |",
            f"| Labeled Voices | {self.labeled_voices_count} |",
            f"| Episode Clusters | {self.episode_clusters_count} |",
            f"| Matched Voices | {self.matched_voices_count} |",
            f"| Match Rate | {self.match_rate*100:.1f}% |",
            f"",
        ]

        if self.labeled_voice_names:
            lines.extend([
                f"### Labeled Voices in Bank",
                f"",
            ])
            for name in self.labeled_voice_names:
                lines.append(f"- {name}")
            lines.append("")

        if self.issues:
            lines.extend([
                f"## Issues Found ({len(self.issues)})",
                f"",
            ])

            for issue in self.issues:
                icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
                lines.extend([
                    f"### {icon} {issue.code}",
                    f"",
                    f"**Severity:** {issue.severity}",
                    f"",
                    f"**Issue:** {issue.message}",
                    f"",
                    f"**Recommendation:** {issue.recommendation}",
                    f"",
                ])
                if issue.details:
                    lines.append(f"<details>")
                    lines.append(f"<summary>Details</summary>")
                    lines.append(f"")
                    lines.append(f"```json")
                    lines.append(json.dumps(issue.details, indent=2))
                    lines.append(f"```")
                    lines.append(f"</details>")
                    lines.append(f"")

        return "\n".join(lines)


def validate_voice_bank(
    ep_id: str,
    show_id: str,
    voice_bank_dir: Optional[Path] = None,
    voice_clusters: Optional[List[Dict[str, Any]]] = None,
    voice_mapping: Optional[Dict[str, str]] = None,
    cast_members: Optional[List[str]] = None,
) -> VoiceBankValidationReport:
    """
    Validate voice bank and generate comprehensive report.

    Args:
        ep_id: Episode identifier
        show_id: Show identifier
        voice_bank_dir: Path to voice bank directory
        voice_clusters: List of voice cluster data from clustering
        voice_mapping: Mapping of cluster IDs to cast names
        cast_members: List of known cast member names

    Returns:
        VoiceBankValidationReport with findings and recommendations
    """
    report = VoiceBankValidationReport(ep_id=ep_id, show_id=show_id)

    # Check voice bank existence
    if voice_bank_dir is None:
        voice_bank_dir = Path("data/voice_bank") / show_id

    report.voice_bank_path = str(voice_bank_dir)
    report.voice_bank_exists = voice_bank_dir.exists()

    if not report.voice_bank_exists:
        report.add_issue(VoiceBankIssue(
            severity="error",
            code="VOICE_BANK_MISSING",
            message=f"Voice bank directory does not exist: {voice_bank_dir}",
            recommendation=(
                "Create the voice bank directory and add labeled voice samples. "
                "Run the voice bank initialization script or manually add voice "
                "embeddings for each cast member."
            ),
        ))
        report.overall_status = "failed"
        return report

    # Check for labeled voices
    labeled_voices = []
    if voice_bank_dir.exists():
        for voice_file in voice_bank_dir.glob("*.json"):
            try:
                with voice_file.open("r") as f:
                    voice_data = json.load(f)
                name = voice_data.get("name") or voice_file.stem
                if voice_data.get("is_labeled", True):  # Default to labeled
                    labeled_voices.append(name)
            except Exception as e:
                LOGGER.debug(f"Failed to read voice file {voice_file}: {e}")

    report.labeled_voices_count = len(labeled_voices)
    report.labeled_voice_names = sorted(labeled_voices)

    if report.labeled_voices_count == 0:
        report.add_issue(VoiceBankIssue(
            severity="error",
            code="NO_LABELED_VOICES",
            message="Voice bank exists but contains no labeled voices",
            recommendation=(
                "Add labeled voice samples for cast members. Each voice entry "
                "should have a 'name' field with the cast member's name and "
                "'is_labeled' set to true. You can create these by running "
                "voice identification on known samples."
            ),
        ))
        report.overall_status = "failed"
        return report

    # Check voice clusters
    if voice_clusters:
        report.episode_clusters_count = len(voice_clusters)

        # Calculate speech duration
        total_duration = 0.0
        speakers = set()
        for cluster in voice_clusters:
            segments = cluster.get("segments", [])
            for seg in segments:
                total_duration += seg.get("end", 0) - seg.get("start", 0)
            if cluster.get("voice_cluster_id"):
                speakers.add(cluster["voice_cluster_id"])

        report.episode_speech_duration_s = round(total_duration, 2)
        report.episode_speakers_found = len(speakers)

    # Check voice mapping
    if voice_mapping:
        matched = sum(1 for v in voice_mapping.values() if v and not v.startswith("unlabeled"))
        report.matched_voices_count = matched
        report.unmatched_clusters_count = report.episode_clusters_count - matched

        if report.episode_clusters_count > 0:
            report.match_rate = matched / report.episode_clusters_count
        else:
            report.match_rate = 0.0

        # Check for issues
        if report.match_rate == 0:
            report.add_issue(VoiceBankIssue(
                severity="error",
                code="NO_VOICE_MATCHES",
                message="No voice clusters matched any labeled voices in the voice bank",
                recommendation=(
                    "This could indicate:\n"
                    "1. Voice embeddings in the bank are from different audio quality/conditions\n"
                    "2. The similarity threshold is too high (try lowering voice_clustering.similarity_threshold)\n"
                    "3. Cast members in this episode are not in the voice bank\n"
                    "4. Audio enhancement changed voice characteristics too much\n\n"
                    "Try: Re-run voice bank training with samples from this show's audio."
                ),
                details={
                    "clusters_found": report.episode_clusters_count,
                    "labeled_voices": report.labeled_voice_names,
                },
            ))
            report.overall_status = "failed"
        elif report.match_rate < 0.5:
            report.add_issue(VoiceBankIssue(
                severity="warning",
                code="LOW_MATCH_RATE",
                message=f"Only {report.match_rate*100:.0f}% of voice clusters matched labeled voices",
                recommendation=(
                    "Consider:\n"
                    "1. Adding more cast members to the voice bank\n"
                    "2. Updating voice embeddings with samples from recent episodes\n"
                    "3. Lowering the similarity threshold for matching\n"
                    "4. Checking if new recurring characters need to be added"
                ),
                details={
                    "matched": matched,
                    "unmatched": report.unmatched_clusters_count,
                    "match_rate": report.match_rate,
                },
            ))
            report.overall_status = "partial"
        else:
            report.overall_status = "success"

        # Check for unmatched major speakers
        if voice_clusters and report.unmatched_clusters_count > 0:
            unmatched_durations = []
            for cluster in voice_clusters:
                cluster_id = cluster.get("voice_cluster_id", "")
                if cluster_id not in voice_mapping or not voice_mapping.get(cluster_id):
                    duration = sum(
                        seg.get("end", 0) - seg.get("start", 0)
                        for seg in cluster.get("segments", [])
                    )
                    if duration > 30:  # More than 30 seconds = significant speaker
                        unmatched_durations.append({
                            "cluster_id": cluster_id,
                            "duration_s": round(duration, 1),
                        })

            if unmatched_durations:
                report.add_issue(VoiceBankIssue(
                    severity="warning",
                    code="UNMATCHED_MAJOR_SPEAKERS",
                    message=f"Found {len(unmatched_durations)} unmatched speakers with significant screen time",
                    recommendation=(
                        "These speakers have substantial dialogue but weren't matched to "
                        "any voice bank entry. Consider adding them to the voice bank if "
                        "they are recurring characters."
                    ),
                    details={"unmatched_speakers": unmatched_durations},
                ))
    else:
        report.add_issue(VoiceBankIssue(
            severity="warning",
            code="NO_MAPPING_DATA",
            message="No voice mapping data provided for validation",
            recommendation=(
                "Voice mapping is required to validate voice bank effectiveness. "
                "Ensure the voice clustering step completed successfully."
            ),
        ))
        report.overall_status = "unknown"

    # Check cast coverage
    if cast_members and report.labeled_voice_names:
        missing_cast = [
            name for name in cast_members
            if name not in report.labeled_voice_names
        ]
        if missing_cast:
            report.add_issue(VoiceBankIssue(
                severity="info",
                code="MISSING_CAST_IN_BANK",
                message=f"{len(missing_cast)} cast members not in voice bank",
                recommendation=(
                    f"Consider adding voice samples for: {', '.join(missing_cast[:5])}"
                    + (f" and {len(missing_cast)-5} more" if len(missing_cast) > 5 else "")
                ),
                details={"missing_cast": missing_cast},
            ))

    # Log results
    if report.overall_status == "success":
        LOGGER.info(
            f"Voice bank validation passed for {ep_id}: "
            f"{report.matched_voices_count}/{report.episode_clusters_count} matched"
        )
    elif report.overall_status == "partial":
        LOGGER.warning(
            f"Voice bank validation partial for {ep_id}: "
            f"{report.matched_voices_count}/{report.episode_clusters_count} matched, "
            f"{len(report.issues)} issues"
        )
    else:
        LOGGER.error(
            f"Voice bank validation failed for {ep_id}: "
            f"{len([i for i in report.issues if i.severity == 'error'])} errors"
        )

    return report


def save_validation_report(
    report: VoiceBankValidationReport,
    output_dir: Optional[Path] = None,
    format: str = "both",  # "json", "markdown", or "both"
) -> List[Path]:
    """
    Save validation report to files.

    Args:
        report: VoiceBankValidationReport to save
        output_dir: Output directory (default: data/reports)
        format: Output format ("json", "markdown", or "both")

    Returns:
        List of paths to saved files
    """
    output_dir = output_dir or Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    base_name = f"voice_bank_validation_{report.ep_id}"

    if format in ("json", "both"):
        json_path = output_dir / f"{base_name}.json"
        with json_path.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)
        saved_files.append(json_path)
        LOGGER.info(f"Saved JSON report: {json_path}")

    if format in ("markdown", "both"):
        md_path = output_dir / f"{base_name}.md"
        md_path.write_text(report.to_markdown())
        saved_files.append(md_path)
        LOGGER.info(f"Saved Markdown report: {md_path}")

    return saved_files
