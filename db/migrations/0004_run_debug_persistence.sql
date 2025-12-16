-- Run-scoped persistence for debug bundle export + UI run selectors.
-- This extends the lightweight v2 artifacts layout with DB-backed tracing.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY,
    ep_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    label text,
    stage_state_json jsonb,
    config_json jsonb
);

CREATE INDEX IF NOT EXISTS idx_runs_ep_id ON runs (ep_id);

CREATE TABLE IF NOT EXISTS job_runs (
    job_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ep_id text NOT NULL,
    job_name text NOT NULL,
    request_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL,
    started_at timestamptz,
    finished_at timestamptz,
    error_text text,
    artifact_index_json jsonb,
    metrics_json jsonb
);

CREATE INDEX IF NOT EXISTS idx_job_runs_run_id ON job_runs (run_id);
CREATE INDEX IF NOT EXISTS idx_job_runs_ep_id ON job_runs (ep_id);
CREATE INDEX IF NOT EXISTS idx_job_runs_job_name ON job_runs (job_name);

CREATE TABLE IF NOT EXISTS identity_locks (
    ep_id text NOT NULL,
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    identity_id text NOT NULL,
    locked boolean NOT NULL DEFAULT true,
    locked_at timestamptz,
    locked_by text,
    reason text,
    PRIMARY KEY (ep_id, run_id, identity_id)
);

CREATE INDEX IF NOT EXISTS idx_identity_locks_ep_run ON identity_locks (ep_id, run_id);

CREATE TABLE IF NOT EXISTS suggestion_batches (
    batch_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ep_id text NOT NULL,
    run_id text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    generator_version text NOT NULL,
    generator_config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    summary_json jsonb
);

CREATE INDEX IF NOT EXISTS idx_suggestion_batches_ep_run ON suggestion_batches (ep_id, run_id);

CREATE TABLE IF NOT EXISTS suggestions (
    suggestion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id uuid NOT NULL REFERENCES suggestion_batches(batch_id) ON DELETE CASCADE,
    ep_id text NOT NULL,
    run_id text NOT NULL,
    type text NOT NULL,
    target_identity_id text,
    suggested_person_id text NOT NULL,
    confidence double precision NOT NULL DEFAULT 0,
    evidence_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    dismissed boolean NOT NULL DEFAULT false,
    dismissed_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_suggestions_batch_id ON suggestions (batch_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_ep_run ON suggestions (ep_id, run_id);

CREATE TABLE IF NOT EXISTS suggestion_applies (
    apply_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id uuid NOT NULL REFERENCES suggestion_batches(batch_id) ON DELETE CASCADE,
    suggestion_id uuid REFERENCES suggestions(suggestion_id) ON DELETE SET NULL,
    ep_id text NOT NULL,
    run_id text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now(),
    applied_by text,
    changes_json jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_suggestion_applies_batch_id ON suggestion_applies (batch_id);
CREATE INDEX IF NOT EXISTS idx_suggestion_applies_ep_run ON suggestion_applies (ep_id, run_id);
