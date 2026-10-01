#!/usr/bin/env python
"""Score the offered behaviours of the bundled observations.

    python examples/predict.py --model <released DriveJev folder or HF repo id>
    python examples/predict.py --backbone /path/Qwen-Drive-1.0-4B --head head.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivejev.policy import DrivingPolicy, Encoder, load_head  # noqa: E402

HERE = Path(__file__).resolve().parent / "observations"

p = argparse.ArgumentParser()
p.add_argument("--model")
p.add_argument("--backbone")
p.add_argument("--head")
args = p.parse_args()
if args.model:
    policy = DrivingPolicy.from_pretrained(args.model)
else:
    encoder = Encoder(args.backbone)
    policy = DrivingPolicy(encoder, {"default": load_head(args.head, encoder.device)[0]})

for case in sorted(HERE.iterdir()):
    record = json.loads((case / "observation.json").read_text())
    for image in record["images"]:
        image["path"] = str(case / image["path"])
    out = policy.predict(record)
    probs = ", ".join(f"{k} {v:.2f}" for k, v in sorted(out["probabilities"].items(), key=lambda kv: -kv[1]))
    print(f"{case.name:18s} -> {out['candidate_id']:18s} [{probs}]  ({out['timing_ms']['total']:.0f} ms, {out['tokens']} tokens)")
    print(f"{'':18s}    reference teacher: {record['reference_teacher']['preferred']}")
