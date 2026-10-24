// Vite configuration for the Conatus website.
// BASE: public path when the site is served from a sub-directory (e.g. GitHub Pages project site:
// BASE=/conatus/ bash web/build.sh). Defaults to "/" (custom domain or Netlify).
import fs from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const base = process.env.BASE ?? "/";
// The package version (single source: python/conatus/__init__.py): the install commands of the site
// are pinned to it (external review 0.1.0, finding 79).
const version = /^__version__ = "([^"]+)"/m.exec(fs.readFileSync(new URL("../python/conatus/__init__.py", import.meta.url), "utf8"))?.[1];
if (!version) throw new Error("vite.config.ts: __version__ not found in python/conatus/__init__.py");

export default defineConfig({
  base,
  plugins: [react()],
  // Build-time constants, identical in the client and server bundles (no hydration mismatch).
  define: {
    __CN_VERSION__: JSON.stringify(version),
    __SITE_URL__: JSON.stringify(process.env.SITE_URL ?? ""),
    __REPO_URL__: JSON.stringify(process.env.REPO_URL ?? ""),
    __WITH_DOCS__: JSON.stringify(process.env.WITH_DOCS === "1"),
    // HASH=1: single-page build with hash routes (#/quickstart), for hosts that serve one page and
    // its files only (a phone preview); see web/build-portable.sh.
    __HASH_ROUTER__: JSON.stringify(process.env.HASH === "1"),
  },
  // Build output of the client lives outside the home (quota): web/dist is only a staging area
  // inside scratch when run through build.sh (see OUT_DIR there).
  build: {
    target: "es2020",
    manifest: true, // read by scripts/prerender.mjs to preload each page's chunk and stylesheet
    cssCodeSplit: true,
    assetsInlineLimit: 2048,
    reportCompressedSize: false,
  },
  ssr: { noExternal: ["react-router"] },
  // Pages import tutorial scripts and their executed outputs from docs/ with ?raw (shown as run).
  server: { host: "127.0.0.1", port: 5173, fs: { allow: [".."] } },
});
