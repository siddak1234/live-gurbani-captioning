-- Corrections feedback loop — Phase 3 (dedicated project "gurbani-captioning").
--
-- Standalone free Supabase project (org siddak1234), separate from Autom8x, so
-- there are no shared-app coexistence concerns. The table lives in `public`,
-- which means the client needs no schema-exposure setting and no Content-Profile
-- header.
--
-- Write path: the app inserts with the project's anon (publishable) key; the RLS
-- INSERT policy allows it. Reads are restricted to service_role (the training /
-- export side). Metadata-only for now — the audio_* columns are reserved and
-- stay NULL until audio storage is added. Reversible via `DROP TABLE`.

create table if not exists public.corrections (
    id                     uuid primary key,                 -- client-generated; idempotency key
    device_id              uuid not null,                    -- anonymous Keychain device id
    session_id             uuid,
    kind                   text not null,
    predicted_shabad_id    integer,
    predicted_line_idx     integer,
    confidence             double precision,
    runner_ups             jsonb not null default '{}'::jsonb,
    ground_truth_shabad_id integer,
    ground_truth_line_idx  integer,
    engine_state           text,
    recent_chunks          jsonb not null default '[]'::jsonb,
    audio_path             text,                             -- reserved (metadata-only for now)
    audio_start            double precision,
    audio_end              double precision,
    app_version            text,
    build_number           text,
    model_version          text,
    schema_version         integer not null,
    export_status          text not null default 'new',
    review_status          text not null default 'unreviewed',
    created_at             timestamptz not null default now(),   -- client event time
    inserted_at            timestamptz not null default now(),   -- server insert time
    constraint corrections_kind_check
        check (kind in ('hardNegPos','softPos','runnerUpEndorsed','lineNudge','retroactive')),
    constraint corrections_export_status_check
        check (export_status in ('new','exported','discarded')),
    constraint corrections_review_status_check
        check (review_status in ('unreviewed','accepted','rejected'))
);

create index if not exists corrections_device_id_idx
    on public.corrections (device_id);
create index if not exists corrections_export_created_idx
    on public.corrections (export_status, created_at);

-- Privileges: anon may INSERT only; service_role full. (anon/authenticated have
-- USAGE on public by default in Supabase.)
grant insert on public.corrections to anon;
grant select, insert, update, delete on public.corrections to service_role;

-- Row-level security. `force` so even the table owner is subject to policies;
-- service_role has BYPASSRLS so the training side still reads everything.
alter table public.corrections enable row level security;
alter table public.corrections force row level security;

-- anon: insert-only, with basic shape validation. No select/update/delete.
create policy corrections_anon_insert
    on public.corrections
    for insert
    to anon
    with check (
        schema_version >= 1
        and device_id is not null
        and kind in ('hardNegPos','softPos','runnerUpEndorsed','lineNudge','retroactive')
    );

-- service_role: explicit full access (belt-and-suspenders alongside BYPASSRLS).
create policy corrections_service_all
    on public.corrections
    for all
    to service_role
    using (true)
    with check (true);
