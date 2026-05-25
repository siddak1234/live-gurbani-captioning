// delete-my-data — right-to-delete for the corrections feedback loop.
//
// Removes every correction row for a given anonymous device_id. Runs with the
// service_role key the Edge runtime injects, so it bypasses RLS (the app's anon
// role has no DELETE grant). The "auth" is knowledge of your own random device
// UUID — not enumerable — which is acceptable for an anonymous, no-account beta.
// Deployed with verify_jwt = false so the app can call it with just its
// device_id (no user session exists by design).
//
// Request:  POST { "device_id": "<uuid>" }
// Response: 200 { "deleted": "<content-range>" } | 4xx/5xx { "error": "..." }

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
  // UUID shape guard (avoids unfiltered deletes).
  if (!/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(deviceId)) {
    return json({ error: "valid device_id required" }, 400);
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  if (!supabaseUrl || !serviceKey) return json({ error: "server misconfigured" }, 500);

  const res = await fetch(
    `${supabaseUrl}/rest/v1/corrections?device_id=eq.${encodeURIComponent(deviceId)}`,
    {
      method: "DELETE",
      headers: {
        apikey: serviceKey,
        Authorization: `Bearer ${serviceKey}`,
        Prefer: "count=exact",
      },
    },
  );

  if (!res.ok) {
    return json({ error: `delete failed: ${res.status}` }, 502);
  }
  // Content-Range looks like "*/<count>" (or "0-N/<count>").
  return json({ deleted: res.headers.get("content-range") ?? "unknown" }, 200);
});
