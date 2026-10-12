#!/usr/bin/env bash
# docs/site/build.sh — build the documentation (Sphinx + MyST) into docs/site/_build/html.
#
#   conda install -c conda-forge --file docs/site/requirements.txt   # once
#   bash docs/site/build.sh                           # incremental
#   CN_DOCS_STRICT=1 bash docs/site/build.sh          # warnings are errors
#
# Tutorials are not executed here: their verified outputs (docs/site/tutorials/outputs/) are
# included as they are. MathJax and the fonts are vendored (docs/site/vendor/fetch.py).
set -euo pipefail
SITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SITE/../.." && pwd)"
OUT="${CN_DOCS_OUT:-$SITE/_build}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
STRICT=(); [ "${CN_DOCS_STRICT:-0}" = 1 ] && STRICT=(-W --keep-going)
python -m sphinx -b html -d "$OUT/doctrees" -j "${CN_DOCS_JOBS:-4}" "${STRICT[@]}" "$SITE" "$OUT/html" "$@"
