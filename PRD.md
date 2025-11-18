# Product Requirements Document — Screenalytics

Screenalytics delivers automated face, voice, and screen-time intelligence for unscripted television, powering catalog metadata, commerce, and fan engagement flows. This PRD captures the CI/promotion process, phased roadmap, risks, acceptance criteria, future enhancements, and key references.

---

## 11. CI / Promotion Flow
1. Develop in `FEATURES/<name>/` (`src/`, `tests/`, `docs/`, `TODO.md`).
2. CI enforces:
   - 30-day TTL
   - Tests + docs present
   - No imports from `FEATURES/**` in production
3. Promote via `tools/promote-feature.py`.
4. Post-promotion CI re-runs end-to-end sample clip.

---

## 12. Phased Roadmap

| Phase | Scope | Key Deliverables |
|-------|-------|------------------|
| **0 – Init** | Repo scaffold | MANIFEST, README, PRD, Architecture, Directory docs |
| **1 – Vision Pipeline** | Detection → Tracking → Embeddings | Accurate track files + metrics |
| **2 – Identity & Facebank** | ArcFace + override UI | Reliable per-cast ID |
| **3 – Audio Fusion** | Diarization + ASR | Speaking-time metrics |
| **4 – UI Integration** | SHOWS + PEOPLE tabs | Unified catalog + featured thumbnails |
| **5 – Agents & Automation** | Codex + MCP | Auto relabel + exports |
| **6 – Production Scale** | Multi-GPU pipeline + dashboard | 100 % episode coverage |

---

## 13. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|---------|-------------|
| **Occlusion / lighting variance** | Misidentifications | Use RetinaFace + quality gating; augment facebank |
| **Storage cost** | ↑ S3 fees | Lifecycle expiry for frames/thumbnails |
| **Audio desync** | Misaligned speech | Sync via ffmpeg timestamps + cross-correlation |
| **Facebank drift** | Identity errors | Periodic re-embedding + QA reports |
| **Agent miswrites** | Doc sprawl | Configured write-policy + CI checks |
| **Human bottleneck** | Slow QA | Cluster/track propagate + batch assign |
| **Legacy imports** | Fragile builds | No `.bak`; pre-commit denylist |

---

## 14. Acceptance Criteria
- ✅ All stages produce valid artifacts (validated schema + checksum).
- ✅ Unit + integration tests pass (coverage > 85 %).
- ✅ Configs documented in `CONFIG_GUIDE.md`.
- ✅ One-hour validation episode processed ≤ 10 min GPU.
- ✅ Exported `screen_time.csv` verified against manual benchmark (± 5 %).
- ✅ Matching row in `ACCEPTANCE_MATRIX.md` marked Accepted for the promoted module.
- ✅ Docs + tests + config + promotion checklist complete.

See `ACCEPTANCE_MATRIX.md` for the detailed acceptance checklist.

---

## 15. Future Enhancements
- Scene-type classification via VLM (e.g., LLaVA / Qwen-VL).
- Emotion / sentiment tagging.
- Multi-camera or split-screen handling.
- Real-time episode analytics stream.
- Dashboard for cross-season trends.

---

## 16. References
- `SOLUTION_ARCHITECTURE.md` — system diagram
- `DIRECTORY_STRUCTURE.md` — repo layout
- `CONFIG_GUIDE.md` — configuration reference
- `ACCEPTANCE_MATRIX.md` — QA checklist
- `FEATURES_GUIDE.md` — promotion policy

---

**Approved by:** TRR Engineering / Analytics / Design / Automation leads
**Next review:** after Phase 1 completion
