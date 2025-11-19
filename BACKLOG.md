# Screenalytics Pipeline — Backlog

**Last Updated:** 2025-11-19

This document tracks deferred tasks and technical debt. These items are not urgent but should be addressed in future maintenance cycles.

---

## 1. FEATURES Documentation Cleanup / Archival

**Priority:** LOW
**Estimated Effort:** 2-3 hours
**Context:** Post-Phase 3 pipeline improvements

### Problem

The `FEATURES/` directory contains experimental and deprecated pipeline implementations that may no longer align with the current pipeline architecture. Some features were superseded by Phase 3 work (profiles, metrics, configs).

### Tasks

1. **Audit `FEATURES/` directory:**
   - Identify which features are still active vs. deprecated
   - Check for code references to FEATURES modules
   - Document status of each feature

2. **Archive deprecated features:**
   - Move obsolete code to `archive/FEATURES/` or delete if fully superseded
   - Update any documentation that references deprecated features
   - Ensure no production code depends on archived modules

3. **Update FEATURES README:**
   - Document which features are still experimental
   - Provide clear migration paths for deprecated features
   - Cross-reference to replacement implementations

### Files to Review

- `FEATURES/*/` - All feature directories
- `docs/FEATURES*.md` - Feature documentation
- Any imports from `FEATURES` in production code

### Success Criteria

- Clear distinction between active experiments and deprecated code
- No broken references to archived features
- Updated documentation reflecting current state

---

## 2. Naming Consistency Pass

**Priority:** LOW
**Estimated Effort:** 3-4 hours
**Context:** Pipeline evolved organically, some naming inconsistencies remain

### Problem

Throughout the pipeline, there are minor naming inconsistencies across:
- **Profile names:** `fast_cpu` vs. `low_power` (aliased but confusing)
- **Metric names:** `short_track_fraction` vs. `short_tracks_fraction` (inconsistent pluralization)
- **Config keys:** Some use `_thresh`, others `_threshold`
- **API fields:** Some use `ep_id`, others `episode_id`

### Tasks

1. **Audit naming patterns:**
   - Catalog all profile names (CLI, API, configs)
   - Catalog all metric names (track_metrics.json, API responses)
   - Catalog all config keys (YAML files)
   - Catalog all API field names

2. **Propose naming standard:**
   - Profile naming: Choose `low_power` or `fast_cpu` (retire alias)
   - Metric naming: Decide on pluralization rules
   - Config keys: Standardize on `_thresh` or `_threshold`
   - API fields: Standardize on `ep_id` or `episode_id`

3. **Incremental migration:**
   - Create compatibility layer (keep old names as aliases)
   - Add deprecation warnings for old names
   - Update documentation to use new names
   - Plan removal of deprecated names for next major version

### Files to Review

- `config/pipeline/*.yaml` - All config files
- `apps/api/routers/jobs.py` - API field names
- `tools/episode_run.py` - CLI parameter names
- `tools/episode_cleanup.py` - CLI parameter names
- `docs/` - All documentation

### Success Criteria

- Consistent naming across code, configs, API, and docs
- Backward compatibility maintained via aliases
- Deprecation warnings guide users to new names

---

## 3. Additional Deferred Items

### 3.1 Performance Profiling

- **Task:** Add runtime profiling to identify bottlenecks in pipeline
- **Why Deferred:** Phase 3 optimizations already addressed major performance issues
- **Priority:** LOW
- **Estimated Effort:** 2-3 hours

### 3.2 Config Validation

- **Task:** Add JSON Schema or Pydantic validation for YAML configs
- **Why Deferred:** Current manual validation is sufficient for now
- **Priority:** LOW
- **Estimated Effort:** 3-4 hours

### 3.3 Metric Visualization

- **Task:** Add Grafana dashboards for pipeline metrics
- **Why Deferred:** Current metric exposure via API is sufficient
- **Priority:** LOW
- **Estimated Effort:** 4-6 hours

---

## How to Use This Backlog

**When to address these items:**
- During quarterly maintenance cycles
- When onboarding new contributors (good starter tasks)
- When a related bug surfaces (address root cause)

**How to prioritize:**
- HIGH: Impacts production or blocks new features
- MEDIUM: Technical debt that will compound
- LOW: Nice-to-have improvements

**Tracking:**
- Move items to active sprint when prioritized
- Update this document as items are completed
- Add new items as they are identified

---

**Maintained by:** Screanalytics Engineering
