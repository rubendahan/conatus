#!/usr/bin/env bash
# web/build.sh — build the static website into web/dist/site/.
#
#   bash web/build.sh                       # typecheck → client + server bundles → prerender
#   DOCS_HTML=<sphinx html dir> bash web/build.sh   # also copy the Sphinx documentation to /docs/
#   BASE=/conatus/ bash web/build.sh        # site served from a sub-path (GitHub Pages)
#
# Needs Node.js >= 20. The site's data (src/generated/*.json) ships with the sources: every number
# on the site comes from the validated documentation of the release.
set -euo pipefail
WEB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WEB"
[ -d node_modules/vite ] || npm ci --no-audit --no-fund
export BASE="${BASE:-/}"
DIST="$WEB/dist"; OUT="$DIST/site"
rm -rf "$DIST"
node scripts/check-tokens.mjs
npx tsc --noEmit -p .
npx vite build --outDir "$DIST/client" --emptyOutDir --logLevel warn
npx vite build --ssr src/entry-server.tsx --outDir "$DIST/server" --emptyOutDir --logLevel warn
[ -n "${DOCS_HTML:-}" ] && export CN_DOCS_HTML="$DOCS_HTML"
node scripts/prerender.mjs "$DIST/client" "$DIST/server"
mkdir -p "$OUT" && cp -r "$DIST/client/." "$OUT/" && rm -rf "$OUT/.vite"
cp ../THIRD_PARTY_NOTICES.md "$OUT/" && mkdir -p "$OUT/licenses" && cp ../licenses/*.txt "$OUT/licenses/"
if [ -n "${DOCS_HTML:-}" ]; then
  mkdir -p "$OUT/docs" && cp -r "$DOCS_HTML/." "$OUT/docs/" && rm -rf "$OUT/docs/.doctrees" "$OUT/docs/.buildinfo"
fi
touch "$OUT/.nojekyll"                     # GitHub Pages: keep _static/ of the Sphinx site
echo "site: $OUT"
