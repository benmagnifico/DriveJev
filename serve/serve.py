#!/usr/bin/env python
"""DriveJev model service: one frozen Qwen-Drive-1.0 encoder, one or more decision heads.

    python serve/serve.py --model path/to/DriveJev-4B --port 9031              # released folder or HF repo id
    python serve/serve.py --backbone /path/Qwen-Drive-1.0-4B --head default=head.pt [--head other=head2.pt]

POST /predict  {arm, episode_id, observation_id, student_obs, candidates, images:[front -0.5, front 0, front_tele 0]}
GET  /model-info
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from drivejev.policy import DrivingPolicy, Encoder, load_head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9031)
    p.add_argument("--model", help="released DriveJev folder or Hugging Face repo id")
    p.add_argument("--backbone", help="Qwen-Drive-1.0 checkpoint (with --head)")
    p.add_argument("--head", action="append", default=[], help="arm=path/to/head.pt|head.safetensors")
    p.add_argument("--schema", action="append", default=[], help="arm=1.0|1.1: observation schema the arm was trained on (default 1.1)")
    p.add_argument("--image-mode", default="normal", choices=["normal", "blank", "no_tele"])
    p.add_argument("--decision-rule", default=None, choices=["argmax", "group"], help="override the decision rule")
    p.add_argument("--log", default=None, help="optional JSONL request log (fresh file)")
    args = p.parse_args()
    heads, info = {}, {}
    if args.model:
        policy = DrivingPolicy.from_pretrained(args.model, image_mode=args.image_mode)
        policy.decision_rule = args.decision_rule or policy.decision_rule
        heads = policy.heads
        info["default"] = {"path": args.model, "sha256": args.model, "observation_schema": policy.observation_schema}
    else:
        if not (args.backbone and args.head):
            raise SystemExit("give --model, or --backbone with at least one --head")
        encoder = Encoder(args.backbone, image_mode=args.image_mode)
        for spec in args.head:
            arm, path = spec.split("=", 1) if "=" in spec else ("default", spec)
            head, _ = load_head(path, encoder.device)
            heads[arm] = head
            info[arm] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "head": head.config, "observation_schema": "1.1"}
        policy = DrivingPolicy(encoder, heads, decision_rule=args.decision_rule or "argmax")
    for spec in args.schema:  # clients (harness, demo) build the observation the arm expects
        arm, schema = spec.split("=", 1)
        info[arm]["observation_schema"] = schema
    lock = threading.Lock()
    log = open(args.log, "x") if args.log else None
    counters = {"requests": 0, "failed": 0}
    meta = {"status": "ready", "schema": "drivejev-online-1.1", "image_mode": args.image_mode, "decision_rule": policy.decision_rule, "arms": info}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, dict(meta, **counters))

        def do_POST(self):
            received = time.perf_counter()
            try:
                source = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                arm = source.get("arm") or next(iter(heads))
                if arm not in heads:
                    raise ValueError(f"unknown arm {arm}")
                record = {"student_obs": source["student_obs"], "candidates": source["candidates"], "images": source["images"], "observation_id": source.get("observation_id")}
                with lock:
                    result = policy.predict(record, arm)
                result["timings_ms"] = dict(result.pop("timing_ms"), service_total=(time.perf_counter() - received) * 1000)
                result["confidence"] = {"value": result["confidence"]}
                result.update(arm=arm, episode_id=source.get("episode_id"), model_version=info[arm]["sha256"])
                counters["requests"] += 1
                if log:
                    log.write(json.dumps({"episode_id": source.get("episode_id"), "observation_id": source.get("observation_id"), "arm": arm,
                                          "candidate_id": result["candidate_id"], "probabilities": result["probabilities"], "timings_ms": result["timings_ms"]}) + "\n")
                    log.flush()
                self.reply(200, result)
            except Exception as exc:
                counters["failed"] += 1
                self.reply(400, {"error": str(exc)[:500]})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()

        def log_message(self, *unused):
            pass

    print(json.dumps({"status": "ready", "port": args.port, "arms": list(heads)}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
