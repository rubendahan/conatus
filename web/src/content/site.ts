// Site-wide constants. Nothing here is a measured number: measured values live in the generated
// data (src/generated/*.json, written by scripts/gen-data.mjs from the repository) or cite a source.
export const SITE = {
  name: "Conatus",
  version: __CN_VERSION__,               // python/conatus/__init__.py, read by vite.config.ts
  license: "BSD-3-Clause",
  year: 2026,
  // Canonical origin used in <link rel=canonical>, the sitemap and Open Graph tags. Set SITE_URL at
  // build time once the site has a public address (e.g. https://conatus-project.org).
  url: __SITE_URL__,
  // The repository is private until v0.1. When public, set REPO_URL at build time and source paths
  // (paper/conatus.tex, results/runs/<run_id>/…) become links; until then they are shown as paths.
  repo: __REPO_URL__,
  // True when the Sphinx reference site is copied under /docs/ (bash web/build.sh --with-docs);
  // otherwise documentation drafts are shown as repository paths, never as dead links.
  docs: __WITH_DOCS__,
} as const;
