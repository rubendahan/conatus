// Minimal static server for the built site (preview, screenshots, Lighthouse), no dependency.
//   node scripts/serve.mjs [dir] [port]      default: scratch/web_build, 4173
// Serves <path>/index.html for directories, 404.html otherwise, like GitHub Pages / Netlify.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(process.argv[2] || path.join(here, "../../scratch/web_build"));
const port = Number(process.argv[3] || 4173);
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".svg": "image/svg+xml", ".webp": "image/webp", ".png": "image/png", ".woff2": "font/woff2", ".woff": "font/woff", ".xml": "application/xml", ".txt": "text/plain", ".pdf": "application/pdf" };
export function serve(root = dir, p = port) {
  const base = path.resolve(root);
  const server = http.createServer((req, res) => {
    let rel;
    try { rel = decodeURIComponent(new URL(req.url, "http://x").pathname); }
    catch { res.writeHead(400); return res.end("bad request"); }      // malformed %-escape
    let f = path.join(base, rel);
    // inside the root itself, not a sibling sharing its prefix (web_build vs web_build_old)
    if (f !== base && !f.startsWith(base + path.sep)) { res.writeHead(403); return res.end(); }
    if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, "index.html");
    let status = 200;
    if (!fs.existsSync(f)) {
      f = path.join(base, "404.html"); status = 404;
      if (process.env.SERVE_LOG) console.error("404", rel);
      if (!fs.existsSync(f)) { res.writeHead(404); return res.end("not found"); } // portable build: no 404.html
    }
    const ext = path.extname(f);
    const cache = /\/assets\//.test(rel) ? "public, max-age=31536000, immutable" : "no-cache";
    // Compress text like every static host does (Netlify, GitHub Pages), so measurements are realistic.
    const gz = /\.(html|js|css|json|svg|xml|txt)$/.test(f) && /gzip/.test(req.headers["accept-encoding"] || "");
    res.writeHead(status, { "content-type": types[ext] || "application/octet-stream", "cache-control": cache, ...(gz ? { "content-encoding": "gzip", vary: "accept-encoding" } : {}) });
    const stream = fs.createReadStream(f);
    (gz ? stream.pipe(zlib.createGzip()) : stream).pipe(res);
  });
  return new Promise((ok) => server.listen(p, "127.0.0.1", () => ok(server)));
}
if (import.meta.url === `file://${process.argv[1]}`) {
  await serve();
  console.log(`serving ${dir} on http://127.0.0.1:${port}/`);
}
