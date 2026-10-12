// Vite configuration for the Conatus website.
// BASE: public path when the site is served from a sub-directory (e.g. GitHub Pages project site:
// BASE=/conatus/ bash web/build.sh). Defaults to "/" (custom domain or Netlify).
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const base = process.env.BASE ?? "/";

export default defineConfig({
  base,
  plugins: [react()],
  // Build-time constants, identical in the client and server bundles (no hydration mismatch).
  define: {
    __SITE_URL__: JSON.stringify(process.env.SITE_URL ?? ""),
    __REPO_URL__: JSON.stringify(process.env.REPO_URL ?? ""),
    __WITH_DOCS__: JSON.stringify(process.env.WITH_DOCS === "1"),
    // HASH=1: single-page build with hash routes (#/quickstart), for hosts that serve one page and
    // its files only (a claude.ai artifact, a phone preview); see web/build-portable.sh.
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
  ssr: { noExternal: ["react-router-dom", "react-router", "@remix-run/router"] },
  // Pages import tutorial scripts and their executed outputs from docs/ with ?raw (shown as run).
  server: { host: "127.0.0.1", port: 5173, fs: { allow: [".."] } },
});
