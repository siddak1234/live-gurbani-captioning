-- Phase 6a — private Storage bucket for correction audio clips.
--
-- Mirrors the corrections table's access model: anon may UPLOAD only (no read,
-- no list); service_role (training/export) reads everything. The bucket is
-- private (not publicly readable). Object key convention:
--   correction-audio/<device_id>/<correction_id>.<ext>   (m4a AAC / caf Opus)
-- The <device_id> prefix isn't RLS-enforced (anon has no identity, same as the
-- table), but keeps objects grouped per device for ingestion + deletion.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
    'correction-audio',
    'correction-audio',
    false,
    5242880,  -- 5 MB ceiling per clip (clips are ~120 KB)
    array['audio/mp4', 'audio/aac', 'audio/x-caf', 'audio/mpeg']
)
on conflict (id) do nothing;

-- anon: insert-only into this bucket. No select/list/update/delete.
create policy "correction_audio_anon_insert"
    on storage.objects
    for insert
    to anon
    with check (bucket_id = 'correction-audio');

-- service_role: full access for ingestion + cleanup.
create policy "correction_audio_service_all"
    on storage.objects
    for all
    to service_role
    using (bucket_id = 'correction-audio')
    with check (bucket_id = 'correction-audio');
