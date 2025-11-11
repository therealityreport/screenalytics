WITH upsert_show AS (
    INSERT INTO show (show_id, slug, title, network, status)
    VALUES (
        '11111111-1111-1111-1111-111111111111'::uuid,
        'show_demo',
        'Demo Show',
        'Screen Network',
        'active'
    )
    ON CONFLICT (slug) DO UPDATE
    SET title = EXCLUDED.title,
        network = EXCLUDED.network,
        status = EXCLUDED.status,
        updated_at = now()
    RETURNING show_id
),
show_row AS (
    SELECT show_id FROM upsert_show
    UNION ALL
    SELECT show_id FROM show WHERE slug = 'show_demo'
    LIMIT 1
),
upsert_season AS (
    INSERT INTO season (season_id, show_id, number, year, meta)
    SELECT
        '22222222-2222-2222-2222-222222222222'::uuid,
        show_id,
        1,
        2023,
        '{}'::jsonb
    FROM show_row
    ON CONFLICT (show_id, number) DO UPDATE
    SET year = EXCLUDED.year,
        updated_at = now()
    RETURNING season_id
),
season_row AS (
    SELECT season_id FROM upsert_season
    UNION ALL
    SELECT season_id FROM season
    WHERE season_id = '22222222-2222-2222-2222-222222222222'::uuid
    LIMIT 1
)
INSERT INTO episode (ep_id, season_id, number, air_date, title, duration_s, meta)
SELECT data.ep_id,
       season_row.season_id,
        data.number,
        data.air_date,
        data.title,
        data.duration_s,
        data.meta
FROM season_row,
(VALUES
    (
        '33333333-3333-3333-3333-333333333333'::uuid,
        1,
        DATE '2023-05-01',
        'Pilot',
        2700,
        jsonb_build_object('code', 'ep_demo')
    ),
    (
        '44444444-4444-4444-4444-444444444444'::uuid,
        2,
        DATE '2023-05-08',
        'Second Shift',
        2700,
        jsonb_build_object('code', 'ep_extra')
    )
) AS data(ep_id, number, air_date, title, duration_s, meta)
ON CONFLICT (season_id, number) DO UPDATE
SET air_date = EXCLUDED.air_date,
    title = EXCLUDED.title,
    duration_s = EXCLUDED.duration_s,
    meta = excluded.meta,
    updated_at = now();

-- Seed a demo person + screen time row to support presence_by_person.
INSERT INTO person (person_id, canonical_name, display_name, aliases)
VALUES (
    '55555555-5555-5555-5555-555555555555'::uuid,
    'demo_person',
    'Screenalytics Demo',
    ARRAY['demo']
)
ON CONFLICT (person_id) DO UPDATE
SET canonical_name = EXCLUDED.canonical_name,
    display_name = EXCLUDED.display_name,
    aliases = EXCLUDED.aliases,
    updated_at = now();

INSERT INTO screen_time (ep_id, person_id, visual_s, speaking_s, both_s, confidence)
VALUES (
    '33333333-3333-3333-3333-333333333333'::uuid,
    '55555555-5555-5555-5555-555555555555'::uuid,
    10,
    5,
    4,
    0.9
)
ON CONFLICT (ep_id, person_id) DO UPDATE
SET visual_s = EXCLUDED.visual_s,
    speaking_s = EXCLUDED.speaking_s,
    both_s = EXCLUDED.both_s,
    confidence = EXCLUDED.confidence;
