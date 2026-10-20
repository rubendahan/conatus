// Conatus Studio service worker: caches the app SHELL only (HTML, scripts, styles, fonts, icons) so
// the app opens instantly and can be installed. Never caches data: /api/*, /ws, /login, /logout and
// any non-GET request always go to the network (run state must never be stale or replayed).
const SHELL = "cn-studio-shell-v1";
const NEVER = [/^\/api\//, /^\/ws/, /^\/login/, /^\/logout/, /^\/relay/, /^\/healthz/, /^\/classic/];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(["/", "/manifest.webmanifest", "/icons/mark-silver.svg"])).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || NEVER.some((r) => r.test(url.pathname))) return;
  const shellAsset = /^\/(assets|fonts|icons)\//.test(url.pathname);
  if (shellAsset) {
    // hashed or fixed files: cache first
    e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(SHELL).then((c) => c.put(req, copy)); }
      return res;
    })));
    return;
  }
  if (req.mode === "navigate") {
    // pages: network first (a redirect to /login must win), the cached shell only when offline
    e.respondWith(fetch(req).catch(() => caches.match("/")));
  }
});
