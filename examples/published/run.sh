#!/usr/bin/env bash
# examples/published/run.sh — run one published case and record what it printed.
#   bash examples/published/run.sh <case> [python]
# Runs examples/published/<case>/run.py in a fresh empty directory and writes, next to the script:
#   output.txt   a header (script sha256, Conatus version, Python, date, wall time, CPU, threads)
#                followed by the exact stdout of the script;
#   result.json  the numbers the script computed (the website draws its figure from them).
# The website shows output.txt next to the script; tests/docs/test_published_cases.py checks that
# every number the site displays is printed in output.txt and that the header's sha256 is the
# script's.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE="${1:?usage: run.sh <case> [python]}"
PY="${2:-python}"
DIR="$HERE/$CASE"
SCRIPT="$DIR/run.py"
[ -f "$SCRIPT" ] || { echo "no script $SCRIPT" >&2; exit 1; }
TMP="$(mktemp)"
WORK="$(mktemp -d)"
cd "$WORK"                                          # a fresh directory: nothing local is imported
t0=$(date +%s)
"$PY" -P "$SCRIPT" > "$TMP" || { rc=$?; cat "$TMP"; echo "run.py failed (exit $rc)" >&2; exit $rc; }
t1=$(date +%s)
cpu="$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
{
  printf '# %s/run.py | script sha256 %s | conatus %s | python %s | %s | %s s | %s | %s threads\n' \
    "$CASE" "$(sha256sum "$SCRIPT" | cut -c1-64)" \
    "$("$PY" -P -c 'import conatus; print(conatus.__version__)')" \
    "$("$PY" -P -c 'import platform; print(platform.python_version())')" \
    "$(date -u +%Y-%m-%d)" "$((t1 - t0))" "$cpu" "${OMP_NUM_THREADS:-1}"
  cat "$TMP"
} > "$DIR/output.txt"
cp result.json "$DIR/result.json"
rm -rf "$TMP" "$WORK"
cat "$DIR/output.txt"
