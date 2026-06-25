-- Least-privilege hardening for public.corrections.
--
-- Supabase grants anon/authenticated broad table privileges on new `public`
-- tables by default. RLS (enabled + forced, with only an anon INSERT policy)
-- already blocks everything except that insert, but we revoke the unused grants
-- so least-privilege also holds at the grant layer. Net effect: anon can INSERT
-- only; reads/updates/deletes remain service_role-only.

revoke all on table public.corrections from anon, authenticated;
grant insert on table public.corrections to anon;
