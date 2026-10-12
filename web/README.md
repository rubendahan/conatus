# Conatus website (`web/`)

The site of Conatus: Vite + React + TypeScript, prerendered to static HTML (one `index.html` per
page, readable without JavaScript, then hydrated). No server and no external service: fonts and
MathJax are self-hosted. The hero animation is a real Kelvin–Helmholtz simulation running in the
browser (WebGL), see `src/sim/README.md`.

```bash
cd web && npm ci
BASE=/conatus/ bash build.sh                   # → web/dist/site/
DOCS_HTML=../docs/site/_build/html BASE=/conatus/ bash build.sh   # with the Sphinx docs under /docs/
node scripts/serve.mjs dist/site               # preview
```

`src/generated/*.json` holds the site's data (validation table, tutorials, citation), produced from
the validated documentation of the release. The gallery images (`public/gallery/`) are frames of
real runs; their scripts are in `gallery/` and `examples/`.
