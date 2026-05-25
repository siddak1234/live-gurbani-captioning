// delete-my-data — right-to-delete for the corrections feedback loop.
//
// Removes every correction ROW and every audio OBJECT for a given anonymous
// device_id. Runs with the service_role key the Edge runtime injects, so it
// bypasses RLS (the app's anon role has no delete grant) and may use the
// Storage API. The "auth" is knowledge of your own random device UUID — not
// enumerable — acceptable for an anonymous, no-account beta. Deployed with
// verify_jwt = false (no user session exists by design).
//
// Request:  POST { "device_id": "<uuid>" }
// Response: 200 { "rows": "<content-range>", "audio_deleted": <n> } | 4xx/5xx { "error": "..." }

const BUCKET = "correction-audio";

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "method not allowed" }, 405);

  let body: { device_id?: unknown };
  try {
    body = await req.json();
  } catch {
    return json({ error: "invalid json" }, 400);
  }

  const deviceId = String(body?.device_id ?? "");
  if (!/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(deviceId)) {
    return json({ error: "valid device_id required" }, 400);
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!supabaseUrl || !serviceKey) return json({ error: "server misconfigured" }, 500);

  const auth = {
    apikey: serviceKey,
    Authorization: `Bearer ${serviceKey}`,
    "content-type": "application/json",
  };

  // 1) Delete metadata rows.
  const rowsRes = await fetch(
    `${supabaseUrl}/rest/v1/corrections?device_id=eq.${encodeURIComponent(deviceId)}`,
    { method: "DELETE", headers: { ...auth, Prefer: "count=exact" } },
  );
  if (!rowsRes.ok) return json({ error: `row delete failed: ${rowsRes.status}` }, 502);

  // 2) Delete audio objects under correction-audio/<device_id>/.
  let audioDeleted = 0;
  const listRes = await fetch(`${supabaseUrl}/storage/v1/object/list/${BUCKET}`, {
    method: "POST",
    headers: auth,
    body: JSON.stringify({ prefix: `${deviceId}/`, limit: 10000 }),
  });
  if (listRes.ok) {
    const items = (await listRes.json()) as Array<{ name: string }>;
    const keys = items.map((o) => `${deviceId}/${o.name}`);
    if (keys.length > 0) {
      await fetch(`${supabaseUrl}/storage/v1/object/${BUCKET}`, {
        method: "DELETE",
        headers: auth,
        body: JSON.stringify({ prefixes: keys }),
      });
      audioDeleted = keys.length;
    }
  }

  return json({ rows: rowsRes.headers.get("content-range") ?? "unknown", audio_deleted: audioDeleted }, 200);
});
