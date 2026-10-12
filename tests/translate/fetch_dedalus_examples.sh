#!/usr/bin/env bash
# The official Dedalus 3.0.5 examples are NOT shipped in the env (dedalus/examples.tar.gz there is
# an empty 58-byte archive). This fetches them from the v3.0.5 tag into scratch (not the repo:
# GPL-3 files of another project, and home space), for tests/translate/.
set -euo pipefail
source "$(dirname "$0")/../../env/env.sh" >/dev/null 2>&1 || true
DEST="$CN_TMP/dedalus_examples"
mkdir -p "$DEST"
cd "$DEST"
[ -d v3.0.5 ] && { echo "$DEST/v3.0.5 already there"; exit 0; }
curl -sL https://github.com/DedalusProject/dedalus/archive/refs/tags/v3.0.5.tar.gz -o d305.tgz
tar xzf d305.tgz --wildcards "dedalus-3.0.5/examples/*"
mv dedalus-3.0.5/examples v3.0.5 && rmdir dedalus-3.0.5 && rm d305.tgz
echo "$DEST/v3.0.5"
