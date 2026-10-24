# Website gallery images

Images of the site's Gallery, each made by a short, real Conatus run (no mock-up), with the two
colormaps of the brand chart (`docs/brand/guidelines.html` §III): sequential **smoke** and signed
**rubric · black · glass** (zero is black). Lab-black background, no axes or text, faint grain.

| File | Role |
|---|---|
| `brand_cmaps.py` | colormaps, resampling to pixels, grain, fluorescein line, WebP output (1600 + 800 px, < 250 kB) and `<slug>.meta.json` |
| `<case>.py` | one script per case: builds the problem with `conatus.expert`, runs it, writes the raw field (`.npz`) and the images in `$CN_OUT` |
| `runs.txt` | `slug run_id` of the runs published on the site, in display order |
| `collect.py` | copies the images to `web/public/gallery/` and generates `manifest.json` (never edit it by hand) |

Regenerate one case (a job, never on the login node):


then put the new `run_id` in `runs.txt` and, on the login node (file copies only):

    source env/env.sh && python web/gallery/collect.py && python ci/archive_results.py

Every manifest entry carries what the pictured run measured (`measured`, built by `collect.py` from
that run's own diagnostics), its colour range (`scale`) and the region it covers (`extent`, in
`fields.json`). Published benchmarks of the same solvers are on the validation page, not on the
pictures.
