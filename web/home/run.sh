#!/usr/bin/env bash
# web/home/run.sh — execute the homepage example and record what it printed.
#   bash web/home/run.sh [python]     (in a compute job: srun -p genoa -c 4 --mem 16G -t 20 bash web/home/run.sh)
# Writes web/home/critical_reynolds.out: a header (script sha256, Conatus version, Python, date) and
# the stdout of the script, exactly. The site and the README show this file next to the script;
# tests/docs/test_home_example.py checks that the header's sha256 is the one of the script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${1:-python}"
SCRIPT="$HERE/critical_reynolds.py"
TMP="$(mktemp)"
cd "$(mktemp -d)"                                   # a fresh directory: nothing local is imported
t0=$(date +%s)
"$PY" -P "$SCRIPT" > "$TMP"
t1=$(date +%s)
{
  printf '# critical_reynolds.py | script sha256 %s | conatus %s | python %s | %s | %s s\n' \
    "$(sha256sum "$SCRIPT" | cut -c1-64)" \
    "$("$PY" -P -c 'import conatus; print(conatus.__version__)')" \
    "$("$PY" -P -c 'import platform; print(platform.python_version())')" \
    "$(date -u +%Y-%m-%d)" "$((t1 - t0))"
  cat "$TMP"
} > "$HERE/critical_reynolds.out"
rm -f "$TMP"
cat "$HERE/critical_reynolds.out"
