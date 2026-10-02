#!/usr/bin/env python
"""Package a DriveJev model for the Hugging Face Hub.

    python tools/export_hf.py --backbone /path/Qwen-Drive-1.0-4B --head head.pt --out DriveJev-4B [--lora adapter_dir] [--copy-backbone]
                              [--release 1.0] [--backbone-license /path/Qwen-Drive-1.0/LICENSE]

Output folder:
    drivejev_config.json   compiler/camera/behaviour contract + head config
    head.safetensors       decision head (FP32)
    backbone/              Qwen-Drive-1.0 checkpoint; with --lora the adapter is merged into the language layers first.
                           --copy-backbone copies the files needed to load the VLM (not the planning experts or the
                           perception head) and adds NOTICE.md (+ LICENSE with --backbone-license).
                           Without --lora and --copy-backbone, the config points at the base model id instead.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivejev.compiler import CAMERAS, COMPILER_VERSION, IMAGE_LAYOUT  # noqa: E402

BEHAVIOURS = ["keep_route_cruise", "stop_at_line", "hold_stop", "proceed_route", "route_turn", "yield_agent", "continue_current", "emergency_brake"]
# Everything QwenDriveForPlanning.from_pretrained, QwenDriveConfig and the tokenizer read (planner-*/ and perception/ are optional extras).
BACKBONE_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "merges.txt", "chat_template.jinja", "chat_template.json", "preprocessor_config.json"]
REQUIRED = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_notice(folder, base_model_id, merged, license_file):
    change = ("A DriveJev LoRA adapter has been merged into the language layers of these weights."
              if merged else "The weights are unmodified. Only the files needed to load the VLM are included "
              "(the planning experts `planner-*/` and the perception head are not).")
    (folder / "NOTICE.md").write_text(
        f"# Qwen-Drive-1.0-4B\n\nThis folder holds the Qwen-Drive-1.0-4B vision-language model "
        f"(https://huggingface.co/{base_model_id}), released under the Apache License 2.0"
        f"{' (see LICENSE)' if license_file else ''}.\n\n{change}\n\n"
        f"`model.safetensors` SHA256: `{sha256(folder / 'model.safetensors')}`\n")
    if license_file:
        shutil.copyfile(license_file, folder / "LICENSE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lora")
    p.add_argument("--copy-backbone", action="store_true")
    p.add_argument("--base-model-id", default="Qwen/Qwen-Drive-1.0-4B")
    p.add_argument("--backbone-license", help="Apache-2.0 LICENSE text to ship next to the backbone weights")
    p.add_argument("--release", help="release version recorded in drivejev_config.json, e.g. 1.0")
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
        write_notice(out / "backbone", args.base_model_id, True, args.backbone_license)
        backbone = "backbone"
    elif args.copy_backbone:
        source = Path(args.backbone)
        missing = REQUIRED - {f for f in BACKBONE_FILES if (source / f).exists()}
        if missing:
            raise SystemExit(f"backbone folder lacks {sorted(missing)}")
        (out / "backbone").mkdir()
        for name in BACKBONE_FILES:
            if (source / name).exists():
                shutil.copy2(source / name, out / "backbone" / name)
        write_notice(out / "backbone", args.base_model_id, False, args.backbone_license)
        backbone = "backbone"
    config = {"format": "drivejev-hf-v1", **({"release": args.release} if args.release else {}), "compiler_version": COMPILER_VERSION,
              "backbone": backbone, "base_model": args.base_model_id, "lora_merged": bool(args.lora),
              "head_file": "head.safetensors", "head": saved["head_config"], "decision_rule": args.decision_rule, "temperature": float(saved.get("temperature", 1.0)),
              "cameras": {k: {"label": v[0], "size": list(v[1])} for k, v in CAMERAS.items()}, "image_layout": [list(x) for x in IMAGE_LAYOUT],
              "behaviours": BEHAVIOURS, "observation_schema": args.observation_schema}
    (out / "drivejev_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"out": str(out), "backbone": backbone, "head": saved["head_config"]["kind"]}))


if __name__ == "__main__":
    main()
