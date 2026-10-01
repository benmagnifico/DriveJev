#!/usr/bin/env python3
"""DriveJev demo web server: the browser app, the JevPilot assets and the pilot backends.

    python demo/server.py [--port 9030] [--model-url http://127.0.0.1:9031] [--kev-url http://127.0.0.1:8014]

Serves the demo and proxies decisions to
  * the DriveJev model service (serve/serve.py)            POST /api/drivejev/predict -> <model-url>/predict
  * cloud Jev (OpenRouter Decisions API)                   POST /api/jev  (only if OPENROUTER_API_KEY is set)
  * a local Jev-compatible server (e.g. Kev)               POST /api/kev  -> <kev-url>/v1/systemone
GET /api/status reports which backends are available. The OpenRouter key is read from the
OPENROUTER_API_KEY environment variable of this process and never sent to the browser.
Standard library only; binds 127.0.0.1.
"""
import argparse
import json
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JEVPILOT = REPO / "third_party/jevpilot"
MAX_BODY = 12 * 1024 * 1024
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"

# URL prefix -> directory. The prefixes mirror the repository layout, so the relative imports
# inside simulator/*.mjs ('../third_party/jevpilot/src/...') resolve to the SAME module URLs the
# app imports, and JevPilot singletons (render-profile.js, simulation.js) load exactly once.
ROUTES = (
    ("/demo/", HERE),
    ("/simulator/", REPO / "simulator"),
    ("/third_party/", REPO / "third_party"),
    ("/vendor/three/", JEVPILOT / "node_modules/three"),
)
PUBLIC = JEVPILOT / "public"  # JevPilot models, textures, draco decoder, brand marks
STATIC_SUFFIXES = {".html", ".js", ".mjs", ".css", ".json", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".glb", ".gltf",
                   ".bin", ".hdr", ".ktx2", ".wasm", ".woff", ".woff2", ".txt"}


def resolve_static(pathname):
    pathname = unquote(pathname)
    if pathname in {"/", "/index.html"}:
        return HERE / "index.html"
    for prefix, root in ROUTES:
        if pathname.startswith(prefix):
            target = (root / pathname[len(prefix):]).resolve()
            break
    else:
        root = PUBLIC
        target = (root / pathname.lstrip("/")).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file() or target.suffix.lower() not in STATIC_SUFFIXES:
        raise FileNotFoundError(pathname)
    return target


def http_json(url, body=None, headers=None, timeout=10.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"error": raw[:300].decode(errors="replace")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=9030)
    parser.add_argument("--model-url", default="http://127.0.0.1:9031", help="DriveJev model service (serve/serve.py)")
    parser.add_argument("--kev-url", default="http://127.0.0.1:8014", help="local Jev-compatible /v1/systemone server")
    parser.add_argument("--jev-model", default="~typesafe/jev-latest", help="OpenRouter Jev model id")
    parser.add_argument("--jev-input-price", type=float, default=0.042, help="USD per million input tokens (cost display)")
    args = parser.parse_args()
    for ext, kind in ((".mjs", "text/javascript"), (".js", "text/javascript"), (".glb", "model/gltf-binary"),
                      (".wasm", "application/wasm"), (".hdr", "application/octet-stream"), (".ktx2", "image/ktx2")):
        mimetypes.add_type(kind, ext)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or None
    active = {"jev": 0, "kev": 0}
    active_lock = threading.Lock()

    def model_info():
        try:
            status, value = http_json(args.model_url + "/model-info", timeout=2)
            if status != 200:
                return {"status": "error", "detail": value}
            return value
        except (OSError, ValueError) as exc:
            return {"status": "unavailable", "detail": f"{exc.__class__.__name__}: model service not reachable"}

    def kev_info():
        try:
            status, value = http_json(args.kev_url + "/v1/models", timeout=1.5)
            return {"available": status == 200, "url": args.kev_url}
        except (OSError, ValueError):
            return {"available": False, "url": args.kev_url}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send(self, status, body, content_type="application/json", cache="no-store"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < MAX_BODY:
                raise ValueError("Invalid request size")
            return json.loads(self.rfile.read(size))

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/status":
                self.send(200, {
                    "drivejev": model_info(),
                    "kev": kev_info(),
                    "jev": {"configured": openrouter_key is not None, "model": args.jev_model,
                            "pricing": {"input_per_million": args.jev_input_price, "output_per_million": 0}},
                })
                return
            if path == "/favicon.ico":
                self.send(204, b"")
                return
            try:
                file = resolve_static(path)
            except FileNotFoundError:
                self.send(404, {"error": "Not found", "path": path})
                return
            kind = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
            long_lived = file.is_relative_to(PUBLIC) or path.startswith("/vendor/")
            self.send(200, file.read_bytes(), kind, "public, max-age=3600" if long_lived else "no-cache")

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                payload = self.body()
            except ValueError as exc:
                self.send(400, {"error": str(exc)})
                return
            if path == "/api/drivejev/predict":
                try:
                    status, value = http_json(args.model_url + "/predict", payload, timeout=8)
                except OSError as exc:
                    status, value = 503, {"error": "DriveJev model service unavailable", "detail": str(exc)}
                except ValueError as exc:
                    status, value = 502, {"error": "DriveJev model service returned invalid JSON", "detail": str(exc)}
                self.send(status, value)
            elif path in {"/api/jev", "/api/kev"}:
                self.decision(path.rsplit("/", 1)[1], payload)
            else:
                self.send(404, {"error": "Unknown endpoint"})

        def decision(self, backend, payload):
            request = payload.get("request") if isinstance(payload, dict) else None
            if not isinstance(request, dict) or not isinstance(request.get("questions"), dict) or \
                    not request["questions"] or "state" not in request:
                self.send(400, {"error": "A prepared {state, questions} decision request is required"})
                return
            if backend == "jev" and openrouter_key is None:
                self.send(503, {"error": "Jev is not configured: set OPENROUTER_API_KEY for the demo server"})
                return
            with active_lock:
                if active[backend] >= 3:
                    self.send(429, {"error": f"Too many active {backend} requests"})
                    return
                active[backend] += 1
            started = time.perf_counter()
            try:
                if backend == "jev":
                    body = dict(request, model=args.jev_model)
                    status, value = http_json(OPENROUTER_URL, body, timeout=10, headers={
                        "Authorization": f"Bearer {openrouter_key}", "X-Title": "DriveJev demo"})
                else:
                    body = dict(request, model=payload.get("model") or request.get("model") or "kev")
                    status, value = http_json(args.kev_url + "/v1/systemone", body, timeout=10)
                hints = {401: "key rejected", 402: "insufficient OpenRouter credit", 429: "rate limited"}
                if status != 200:
                    message = value.get("error") if isinstance(value, dict) else None
                    if isinstance(message, dict):
                        message = message.get("message")
                    self.send(status, {"error": f"{backend} HTTP {status}" + (f" ({hints[status]})" if status in hints else "") +
                                       (f": {message}" if message else "")})
                    return
                value["latency_ms"] = round((time.perf_counter() - started) * 1000)
                value["request_bytes"] = len(json.dumps(body))
                self.send(200, value)
            except (OSError, ValueError) as exc:
                self.send(504 if "timed out" in str(exc) else 502, {"error": f"{backend} unreachable: {exc.__class__.__name__}"})
            finally:
                with active_lock:
                    active[backend] -= 1

        def log_message(self, *unused):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(json.dumps({"status": "ready", "url": f"http://127.0.0.1:{args.port}/", "model_url": args.model_url,
                      "jev_configured": openrouter_key is not None}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
