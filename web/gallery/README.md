# Website gallery films and images

The Gallery shows one short looping film per case (`gallery_films.py`), each the run of the case's
problem with its state kept at regular intervals; a case without a film keeps its still image.
Films and stills use the two
colormaps of the brand chart (`docs/brand/guidelines.html` §III): sequential **smoke** and signed
**rubric · black · glass** (zero is black). Lab-black background, no axes or text, faint grain.

| File | Role |
|---|---|
| `brand_cmaps.py` | colormaps, resampling to pixels, grain, fluorescein line, WebP output (1600 + 800 px, < 250 kB) and `<slug>.meta.json` |
| `<case>.py` | one script per case: builds the problem with `conatus.expert`, runs it, writes the raw field (`.npz`) and the images in `$CN_OUT` |
| `runs.txt` | `slug run_id` of the still images, in display order (every case of the Gallery) |
| `gallery_films.py` | one film per case: reruns the case's problem (same parameters and numerics, or a change said in the caption), one fixed colour range per film, WebM + MP4 at the desktop width and at 800 px, two WebP posters and `<slug>.film.json` in `$CN_OUT` |
| `gallery_films_private.py` | the films of the development-only cases (pipe modes, shell, channel), run by `gallery_films.py --case` |
| `film_runs.txt` | `slug run_id` of the film runs published on the site |
| `collect.py` | copies the films (or, without a film, the images) to `web/public/gallery/` and generates `manifest.json` (never edit it by hand) |

Make the films (a job, never on the login node), several cases side by side, `case:threads`:


then put the new `run_id` of each film in `film_runs.txt` and collect as below.

Regenerate one still (a job, never on the login node):


then put the new `run_id` in `runs.txt` and, on the login node (file copies only):

    source env/env.sh && python web/gallery/collect.py && python ci/archive_results.py

Every manifest entry carries what the pictured run measured (`measured`, built by `collect.py` from
that run's own diagnostics), its colour range (`scale`) and the region it covers (`extent`, in
`fields.json`). Published benchmarks of the same solvers are on the validation page, not on the
pictures.
