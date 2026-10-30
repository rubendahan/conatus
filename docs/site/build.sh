#!/usr/bin/env bash
# docs/site/build.sh — build the documentation (Sphinx + MyST) into docs/site/_build/html.
#
#   bash docs/site/build.sh --setup                   # once: creates the docs environment, then builds
#   bash docs/site/build.sh                           # incremental
#   CN_DOCS_STRICT=1 bash docs/site/build.sh          # warnings are errors
#
# --setup creates a virtual environment in docs/site/_build/env and installs docs/site/requirements.txt
# with pip. Later builds use that environment; without it, the build uses `python` from PATH.
# Tutorials are not executed here: their verified outputs (docs/site/tutorials/outputs/) are
# included as they are. MathJax and the fonts are vendored (docs/site/vendor/fetch.py).
set -euo pipefail
SITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SITE/../.." && pwd)"
OUT="${CN_DOCS_OUT:-$SITE/_build}"
ENV="$OUT/env"
if [ "${1:-}" = "--setup" ]; then
    shift
    [ -x "$ENV/bin/python" ] || python3 -m venv "$ENV"
    "$ENV/bin/python" -m pip install -r "$SITE/requirements.txt"
fi
PY=python; [ -x "$ENV/bin/python" ] && PY="$ENV/bin/python"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
STRICT=(); [ "${CN_DOCS_STRICT:-0}" = 1 ] && STRICT=(-W --keep-going)
"$PY" -m sphinx -b html -d "$OUT/doctrees" -j "${CN_DOCS_JOBS:-4}" "${STRICT[@]}" "$SITE" "$OUT/html" "$@"
