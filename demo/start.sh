#!/usr/bin/env bash
# DriveJev demo launcher: model service (serve/serve.py, default :9031) + web server (default :9030).
#
#   DRIVEJEV_MODEL=<released folder or HF repo id> bash demo/start.sh
#   DRIVEJEV_BACKBONE=<Qwen-Drive-1.0-4B dir> DRIVEJEV_HEAD=<head.pt> bash demo/start.sh
#   bash demo/start.sh            # reuse a model service that already answers, or run without one
#
# Optional: PYTHON (interpreter, default python3/python on PATH), WEB_PORT, MODEL_PORT,
# OPENROUTER_API_KEY (enables the cloud Jev pilot), KEV_URL. Ctrl+C stops what this script started.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/.." && pwd)
PYTHON=${PYTHON:-$(command -v python3 || command -v python || true)}
[[ -n "$PYTHON" ]] || { echo "No python found on PATH; set PYTHON=/path/to/python" >&2; exit 1; }
WEB_PORT=${WEB_PORT:-9030}
MODEL_PORT=${MODEL_PORT:-9031}
MODEL_URL="http://127.0.0.1:$MODEL_PORT"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

answers() { "$PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' "$1" >/dev/null 2>&1; }

if answers "$MODEL_URL/model-info"; then
  echo "Using the DriveJev model service already running on :$MODEL_PORT"
else
  serve=()
  if [[ -n "${DRIVEJEV_MODEL:-}" ]]; then
    serve=(--model "$DRIVEJEV_MODEL")
  elif [[ -n "${DRIVEJEV_BACKBONE:-}" && -n "${DRIVEJEV_HEAD:-}" ]]; then
    serve=(--backbone "$DRIVEJEV_BACKBONE" --head "default=$DRIVEJEV_HEAD")
  fi
  if ((${#serve[@]})); then
    echo "Starting the DriveJev model service on :$MODEL_PORT (the DriveJev pilot is available once it has loaded)"
    (cd "$REPO" && exec "$PYTHON" serve/serve.py "${serve[@]}" --port "$MODEL_PORT") &
    pids+=($!)
  else
    echo "No model service on :$MODEL_PORT and neither DRIVEJEV_MODEL nor DRIVEJEV_BACKBONE+DRIVEJEV_HEAD is set." >&2
    echo "Starting the web server only: the reference teacher can drive, DriveJev shows 'model service offline'." >&2
  fi
fi

echo "Open http://127.0.0.1:$WEB_PORT/"
"$PYTHON" "$HERE/server.py" --port "$WEB_PORT" --model-url "$MODEL_URL" ${KEV_URL:+--kev-url "$KEV_URL"}
