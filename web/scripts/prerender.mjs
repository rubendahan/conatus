// Prerender every route to static HTML (SEO, instant first paint), then write the search index,
// the sitemap and robots.txt. Input: the client build (index.html + assets) and the server bundle.
//   node scripts/prerender.mjs <clientDir> <serverDir>
// Each page becomes <route>/index.html (and 404.html), so any static host serves it directly.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [clientDir, serverDir] = process.argv.slice(2);
if (!clientDir || !serverDir) { console.error("usage: prerender.mjs <clientDir> <serverDir>"); process.exit(2); }
const template = fs.readFileSync(path.join(clientDir, "index.html"), "utf8");
const serverEntry = fs.readdirSync(serverDir).find((f) => /^entry-server\.(m?js)$/.test(f));
const { render, allRoutes, redirects } = await import(pathToFileURL(path.join(serverDir, serverEntry)).href);

// Vite manifest: preload each page's own chunk and stylesheet, so the prerendered page is styled
// before any JavaScript runs (page CSS is code-split with the page) and hydrates without a waterfall.
const manifest = JSON.parse(fs.readFileSync(path.join(clientDir, ".vite", "manifest.json"), "utf8"));
const entryKey = Object.keys(manifest).find((k) => manifest[k].isEntry);
function assetsFor(src) {
  const js = new Set(), css = new Set(), seen = new Set();
  const walk = (key) => {
    const m = manifest[key];
    if (!m || seen.has(key)) return;
    seen.add(key);
    if (key !== entryKey) js.add(m.file);
    (m.css || []).forEach((c) => css.add(c));
    (m.imports || []).forEach(walk);
  };
  walk(src);
  (manifest[entryKey].css || []).forEach((c) => css.delete(c)); // already linked by index.html
  return { js: [...js], css: [...css] };
}

// Preload the two faces every page paints first (the display title and the body text), so the
// largest text renders in its final font without a late swap.
const fontFiles = fs.readdirSync(path.join(clientDir, "assets"));
const fontPreload = [/^im-fell-english-latin-400-normal-.*\.woff2$/, /^ibm-plex-sans-latin-400-normal-.*\.woff2$/]
  .map((re) => fontFiles.find((f) => re.test(f))).filter(Boolean)
  .map((f) => `<link rel="preload" as="font" type="font/woff2" crossorigin href="BASE_assets/${f}">`);

const SITE_URL = (process.env.SITE_URL || "").replace(/\/$/, "");
const BASE = (process.env.BASE || "/").replace(/\/?$/, "/");
// The social card image is announced only when the build ships one (public/og.png): a missing
// og:image is a broken preview on every link shared.
const HAS_OG = fs.existsSync(path.join(clientDir, "og.png"));

const esc = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const decode = (s) => s.replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n));
const textOf = (html) => decode(html.replace(/<(script|style|svg)[\s\S]*?<\/\1>/g, " ").replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();

const searchDocs = [];
let pages = 0;
for (const r of allRoutes) {
  const html = await render(r.path);
  const title = r.path === "/" ? r.title : `${r.title} · Conatus`;
  const url = SITE_URL ? SITE_URL + (r.path === "/" ? "/" : r.path + "/") : "";
  const head = [
    `<title>${esc(title)}</title>`,
    `<meta name="description" content="${esc(r.description)}" />`,
    url && r.path !== "/404" ? `<link rel="canonical" href="${url}" />` : "",
    `<meta property="og:type" content="website" />`,
    `<meta property="og:title" content="${esc(title)}" />`,
    `<meta property="og:description" content="${esc(r.description)}" />`,
    SITE_URL && HAS_OG ? `<meta property="og:image" content="${SITE_URL}/og.png" />` : "",
    `<meta name="twitter:card" content="${HAS_OG ? "summary_large_image" : "summary"}" />`,
    r.path === "/404" ? `<meta name="robots" content="noindex" />` : "",
  ].filter(Boolean).join("\n    ");
  const pre = assetsFor(r.src);
  const preload = [
    ...fontPreload.map((l) => l.replace("BASE_", BASE)),
    ...pre.css.map((f) => `<link rel="stylesheet" crossorigin href="${BASE}${f}">`),
    ...pre.js.map((f) => `<link rel="modulepreload" crossorigin href="${BASE}${f}">`),
  ].join("\n    ");
  let out = template.replace("<!--head-->", head).replace("</head>", `  ${preload}\n  </head>`).replace("<!--app-->", html);
  // Inline the stylesheets (site + page, ≈ 10 kB compressed): the first paint needs no extra round trip.
  out = out.replace(/<link rel="stylesheet" crossorigin href="([^"]+)">/g, (_, href) => {
    let css = fs.readFileSync(path.join(clientDir, href.slice(BASE.length)), "utf8");
    // With a relative base ("./", portable build) the CSS urls are relative to assets/: once the
    // stylesheet is inlined in the page they must be rebased onto it.
    if (BASE.startsWith(".")) css = css.replace(/url\(\.\/([^)]+)\)/g, `url(${BASE}assets/$1)`);
    return `<style>${css.replace(/<\/style/gi, "<\\/style")}</style>`;
  });
  const file = r.path === "/404" ? path.join(clientDir, "404.html")
    : r.path === "/" ? path.join(clientDir, "index.html") : path.join(clientDir, r.path.slice(1), "index.html");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, out);
  pages++;

  // Search index: one document per <h2>/<h3> section of <main>, plus the page lead.
  if (r.path === "/404") continue;
  const main = (html.match(/<main[^>]*>([\s\S]*)<\/main>/) || [, html])[1];
  const parts = main.split(/(?=<h[23][^>]*>)/);
  parts.forEach((part, i) => {
    const hm = part.match(/^<h([23])[^>]*>([\s\S]*?)<\/h\1>/);
    const idm = hm && hm[2].match(/id="([^"]+)"/);
    const heading = hm ? textOf(hm[2]) : r.nav ?? r.title;
    const text = textOf(hm ? part.slice(hm[0].length) : part);
    if (!text && !hm) return;
    searchDocs.push({ id: `${r.path}#${idm ? idm[1] : "s" + i}`, path: r.path, page: r.nav ?? r.title, heading, text: text.slice(0, 4000) });
  });
}
// The documentation in the same index (W1: the site search found 0 of the Sphinx pages). With
// CN_DOCS_HTML (web/build.sh --with-docs), every page of the built Sphinx site is read: one document
// per <section id> of its <article role="main">, under the page title, text cut to DOC_TEXT
// characters (the API pages are many; the snippet only has to say where to look). The index, the
// genindex/search pages, highlighted sources (_modules) and raw sources (_sources) are skipped.
const DOC_TEXT = 700;
let docDocs = 0;
const docsDir = process.env.CN_DOCS_HTML;
if (docsDir && fs.existsSync(path.join(docsDir, "index.html"))) {
  const skip = /^(_modules|_sources|_static|_images|_downloads|_sphinx_design_static)\/|^(genindex|search|py-modindex)\.html$/;
  const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(dir, e.name)) : e.name.endsWith(".html") ? [path.join(dir, e.name)] : []);
  for (const file of walk(docsDir).sort()) {
    const rel = path.relative(docsDir, file).split(path.sep).join("/");
    if (skip.test(rel)) continue;
    const html = fs.readFileSync(file, "utf8");
    const art = (html.match(/<article role="main"[^>]*>([\s\S]*?)<\/article>/) || [])[1];
    if (!art) continue;
    const title = textOf((html.match(/<title>([\s\S]*?)<\/title>/) || [, rel])[1]).replace(/\s*[-–—|]\s*Conatus.*$/, "");
    const page = `Docs · ${title}`;
    const docPath = "/docs/" + rel;
    art.split(/(?=<section id=")/).forEach((part, i) => {
      const idm = part.match(/^<section id="([^"]+)"/);
      const hm = part.match(/<h([1-4])[^>]*>([\s\S]*?)<\/h\1>/);
      const heading = hm ? textOf(hm[2]).replace(/[¶#]\s*$/, "").trim() : title;
      let body = hm ? part.slice(part.indexOf(hm[0]) + hm[0].length) : part;
      body = body.split(/<section id="/)[0];          // this section's own text, not its children
      const text = textOf(body).slice(0, DOC_TEXT);
      if (!text && !hm) return;
      if (!idm && i > 0) return;
      searchDocs.push({ id: `${docPath}#${idm ? idm[1] : "top"}`, path: docPath, page, heading, text });
      docDocs++;
    });
  }
}
fs.writeFileSync(path.join(clientDir, "search-index.json"), JSON.stringify(searchDocs));

// Old addresses (routes.ts redirects): a static page that sends the browser to the new one.
for (const r of redirects) {
  // relative base (portable build): relative to the old folder; else under the site's base path
  const href = BASE.startsWith(".") ? path.posix.relative(r.from, r.to) + "/" : BASE + r.to.slice(1) + "/";
  const file = path.join(clientDir, r.from.slice(1), "index.html");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Moved · Conatus</title>
<meta name="robots" content="noindex"><link rel="canonical" href="${SITE_URL ? SITE_URL + r.to + "/" : href}">
<meta http-equiv="refresh" content="0; url=${href}"></head>
<body><p>This page moved to <a href="${href}">${r.to}</a>.</p><script>location.replace(${JSON.stringify(href)} + location.hash)</script></body></html>
`);
}

const today = new Date().toISOString().slice(0, 10);
// SITE_URL is the full address of the site, base path included (https://host/conatus): the base
// is added only when there is no SITE_URL (it was added twice: /conatus/conatus/ in every entry).
const locBase = SITE_URL || (BASE === "/" || BASE.startsWith(".") ? "" : BASE.replace(/\/$/, ""));
const locs = allRoutes.filter((r) => r.path !== "/404").map((r) => `  <url><loc>${locBase}${r.path === "/" ? "/" : r.path + "/"}</loc><lastmod>${today}</lastmod></url>`);
fs.writeFileSync(path.join(clientDir, "sitemap.xml"), `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${locs.join("\n")}\n</urlset>\n`);
// robots.txt: search engines are kept out unless the build sets INDEXABLE=1 (the published site does).
fs.writeFileSync(path.join(clientDir, "robots.txt"), process.env.INDEXABLE === "1"
  ? `User-agent: *\nAllow: /\n${SITE_URL ? `Sitemap: ${SITE_URL}/sitemap.xml\n` : ""}` : "User-agent: *\nDisallow: /\n");
console.log(`prerendered ${pages} pages, ${searchDocs.length} search documents (${docDocs} from the documentation)`);
