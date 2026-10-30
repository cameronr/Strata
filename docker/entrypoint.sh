#!/bin/sh
# The Lamina container's start (Dockerfile): setup.py --no-build prepares MODEL in /data (the downloads and the pack
# once; later starts check them), then lamina-server serves it on port 8080 to other machines, with API_KEY.
set -eu
cd /opt/lamina
if [ -z "${MODEL:-}" ]; then
  echo "set MODEL: iq3_s, ud-q4_k_xl or qwen36-ud-q4_k_xl (docker run -e MODEL=...)" >&2
  exit 2
fi
if [ -z "${API_KEY:-}" ]; then
  echo "set API_KEY: the server answers beyond this container only with a key (docker run -e API_KEY=...)" >&2
  exit 2
fi
set -- --no-build --model "$MODEL" --data-dir /data --vision "${VISION:-on}"
[ -n "${CONTEXT:-}" ] && set -- "$@" --context "$CONTEXT"
[ -n "${MAIN_GPU:-}" ] && set -- "$@" --main-gpu "$MAIN_GPU"
.venv/bin/python setup.py "$@"
# the key reaches the server through its environment, not its command line
LAMINA_API_KEY="$API_KEY"
export LAMINA_API_KEY
exec build/lamina-server --config "lamina-$MODEL.json" --host 0.0.0.0 --port 8080
