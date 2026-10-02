#!/usr/bin/env python
"""Package a DriveJev model for the Hugging Face Hub.

    python tools/export_hf.py --backbone /path/Qwen-Drive-1.0-4B --head head.pt --out DriveJev-4B [--lora adapter_dir] [--copy-backbone]

Output folder:
    drivejev_config.json   compiler/camera/behaviour contract + head config
    head.safetensors       decision head (FP32)
    backbone/              Qwen-Drive-1.0 checkpoint; with --lora the adapter is merged into the language layers first.
                           Without --lora and --copy-backbone, the config points at the base model id instead.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivejev.compiler import CAMERAS, COMPILER_VERSION, IMAGE_LAYOUT  # noqa: E402

BEHAVIOURS = ["keep_route_cruise", "stop_at_line", "hold_stop", "proceed_route", "route_turn", "yield_agent", "continue_current", "emergency_brake"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lora")
    p.add_argument("--copy-backbone", action="store_true")
    p.add_argument("--base-model-id", default="Qwen/Qwen-Drive-1.0-4B")
    p.add_argument("--decision-rule", default="argmax", choices=["argmax", "group"])
    p.add_argument("--observation-schema", default="1.1", choices=["1.0", "1.1"], help="student observation the head was trained on")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.head, map_location="cpu", weights_only=False)
    from safetensors.torch import save_file
    save_file({k: v.float().contiguous() for k, v in saved["state_dict"].items()}, str(out / "head.safetensors"))
    backbone = args.base_model_id
    if args.lora:
        from peft import PeftModel
        from qwen_drive import QwenDriveForPlanning
        container = QwenDriveForPlanning.from_pretrained(args.backbone, dtype=torch.bfloat16, device_map={"": "cpu"})
        container.vlm.model = PeftModel.from_pretrained(container.vlm.model, args.lora).merge_and_unload()
        container.save_pretrained(out / "backbone")
        AutoTokenizer = __import__("transformers").AutoTokenizer
        AutoTokenizer.from_pretrained(args.backbone).save_pretrained(out / "backbone")
        backbone = "backbone"
    elif args.copy_backbone:
        shutil.copytree(args.backbone, out / "backbone")
        backbone = "backbone"
    config = {"format": "drivejev-hf-v1", "compiler_version": COMPILER_VERSION, "backbone": backbone, "lora_merged": bool(args.lora),
              "head_file": "head.safetensors", "head": saved["head_config"], "decision_rule": args.decision_rule, "temperature": float(saved.get("temperature", 1.0)),
              "cameras": {k: {"label": v[0], "size": list(v[1])} for k, v in CAMERAS.items()}, "image_layout": [list(x) for x in IMAGE_LAYOUT],
              "behaviours": BEHAVIOURS, "observation_schema": args.observation_schema}
    (out / "drivejev_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"out": str(out), "backbone": backbone, "head": saved["head_config"]["kind"]}))


if __name__ == "__main__":
    main()
