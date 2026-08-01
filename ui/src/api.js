const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  stats: () => request("/stats"),
  capabilities: () => request("/capabilities"),
  search: (body) => request("/search", { method: "POST", body: JSON.stringify(body) }),
  patent: (id) => request(`/patents/${encodeURIComponent(id)}`),
};
