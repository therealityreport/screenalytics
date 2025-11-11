CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS show (
    show_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    slug text NOT NULL UNIQUE,
    title text NOT NULL,
    network text,
    status text NOT NULL DEFAULT 'draft',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS season (
    season_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    show_id uuid NOT NULL REFERENCES show(show_id) ON DELETE CASCADE,
    number int NOT NULL,
    year int,
    meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (show_id, number)
);

CREATE TABLE IF NOT EXISTS episode (
    ep_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    season_id uuid NOT NULL REFERENCES season(season_id) ON DELETE CASCADE,
    number int NOT NULL,
    air_date date,
    title text,
    duration_s int,
    meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (season_id, number)
);

CREATE TABLE IF NOT EXISTS person (
    person_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name text NOT NULL,
    display_name text NOT NULL,
    aliases text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS screen_time (
    ep_id uuid NOT NULL REFERENCES episode(ep_id) ON DELETE CASCADE,
    person_id uuid NOT NULL REFERENCES person(person_id) ON DELETE CASCADE,
    visual_s double precision NOT NULL DEFAULT 0,
    speaking_s double precision NOT NULL DEFAULT 0,
    both_s double precision NOT NULL DEFAULT 0,
    confidence double precision NOT NULL DEFAULT 0,
    PRIMARY KEY (ep_id, person_id)
);
