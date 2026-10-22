#!/usr/bin/env bash
# Starts the Strata server (OpenAI /v1/chat/completions, Anthropic /v1/messages, the web page) on Nemotron 3.5 Super
# (strata-nh-engine).  Paths in the config are relative to NH_ROOT, the directory that holds build/, models/ and pack/
# (default: three levels above this script, the port's work directory).
#   NH_ROOT=<dir>  NH_PYTHON=<python>  NH_CONFIG=<json>  PORT=8095  HOST=127.0.0.1  nemotron/run-nemotron.sh [server options]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATA="$(cd "$HERE/.." && pwd)"
NH_ROOT="${NH_ROOT:-$(cd "$STRATA/../.." && pwd)}"
CFG="${NH_CONFIG:-$HERE/strata-nemotron.json}"
PY="${NH_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$NH_ROOT/env/dl/bin/python3" ]; then PY="$NH_ROOT/env/dl/bin/python3"; else PY=python3; fi
fi
cd "$NH_ROOT"
mkdir -p logs
for f in build/strata/strata-nh-engine pack/nemotron/tokenizer/tokenizer.json; do
  [ -e "$f" ] || { echo "run-nemotron: $NH_ROOT/$f is missing" >&2; exit 1; }
done
exec "$PY" "$STRATA/serve/server.py" --engine strata --config "$CFG" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8095}" "$@"
