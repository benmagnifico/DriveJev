"""DriveJev prompt compiler.

Prompt = [front t-0.5 s] [front t] [front tele t] + whitelisted student_obs JSON + 2–12
semantic candidates (IDs never in text, canonical semantic order) + "Decision:".
Hidden states read by the head: the decision token, the last token of every candidate,
and the last token of the student-state block.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .ordering import canonicalize

COMPILER_VERSION = "drivejev-1.0-wide2-tele1"
CAMERAS = {
    # camera -> (label, target size (w, h))
    "front": ("<FRONT VIEW>", (640, 384)),
    "front_tele": ("<FRONT TELE VIEW 15deg>", (384, 224)),
}
IMAGE_LAYOUT = (("front", -0.5), ("front", 0.0), ("front_tele", 0.0))
# Only these fields are serialized. Unknown fields are rejected, so privileged teacher
# or evaluation data (signal colours, rules, other agents' plans) cannot leak in.
STUDENT_FIELDS = {
    "ego": {"speed_mps", "acceleration_mps2", "steering", "route_offset_m", "heading_error_deg", "stationary_s"},
    "nav": {"remaining_m", "next_turn", "turn_distance_m", "speed_limit_mps", "junction_control", "stop_line_ahead_m", "stop_completed"},
    "maneuver": {"action_type", "target_id", "elapsed_s", "progress_m", "status", "feedback"},
    "recent_actions": {"action_type", "age_s"},
    "traffic": {"lead", "hazard", "junction"},
}
TRAFFIC_FIELDS = {
    "lead": {"gap_m", "speed_mps"},
    "hazard": {"type", "in_s", "distance_m", "side"},
    "junction": {"vehicles_inside", "cross_approaching", "pedestrians_crossing", "earlier_arrivals"},
}


@dataclass
class CompiledDecision:
    model_inputs: dict
    decision_position: int
    candidate_positions: list
    state_position: int
    candidate_ids: list
    action_types: list
    targets: torch.Tensor | None
    metadata: dict

    def to(self, device, dtype=torch.bfloat16):
        inputs = {k: v.to(device=device, dtype=dtype if k == "pixel_values" else v.dtype) for k, v in self.model_inputs.items()}
        return CompiledDecision(inputs, self.decision_position, self.candidate_positions, self.state_position, self.candidate_ids,
                                self.action_types, None if self.targets is None else self.targets.to(device), self.metadata)


def validate_student(obs):
    unknown = set(obs) - set(STUDENT_FIELDS)
    if unknown:
        raise ValueError(f"Unapproved student fields: {sorted(unknown)}")
    for key, fields in STUDENT_FIELDS.items():
        if key not in obs:
            continue
        sections = obs[key] if isinstance(obs[key], list) else [obs[key]]
        for section in sections:
            bad = set(section) - fields
            if bad:
                raise ValueError(f"Unapproved student {key} fields: {sorted(bad)}")
    for key, value in (obs.get("traffic") or {}).items():
        if value is not None and set(value) - TRAFFIC_FIELDS[key]:
            raise ValueError(f"Unapproved traffic.{key} fields: {sorted(set(value) - TRAFFIC_FIELDS[key])}")
    return json.dumps(obs, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def load_image(image, root=None):
    from PIL import Image
    if image.get("png_base64"):
        raw = base64.b64decode(image["png_base64"])
    else:
        path = Path(image["path"])
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if image.get("sha256") and image["sha256"] != digest:
        raise ValueError("Image bytes differ from the recorded SHA256")
    return Image.open(io.BytesIO(raw)).convert("RGB"), digest


class DriveCompiler:
    def __init__(self, model_path, max_tokens=2048, image_root=None, image_mode="normal"):
        from transformers import AutoTokenizer
        from qwen_drive import QwenDriveConfig, QwenDriveProcessor
        self.config = QwenDriveConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = QwenDriveProcessor(self.tokenizer, self.config)
        self.max_tokens = max_tokens
        self.image_root = image_root
        if image_mode not in {"normal", "blank", "no_tele"}:
            raise ValueError("image_mode must be normal, blank or no_tele")
        self.image_mode = image_mode

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def compile_record(self, record):
        from PIL import Image
        from qwen_drive import CameraFrame
        p = self.processor
        candidates, layout = canonicalize(record["candidates"])
        ids = [str(c["candidate_id"]) for c in candidates]
        images = record.get("images") or []
        layout_found = tuple((im.get("camera", "front"), float(im.get("relative_time", 0))) for im in images)
        if layout_found != IMAGE_LAYOUT:
            raise ValueError(f"Expected image layout {IMAGE_LAYOUT}, got {layout_found}")
        prompt = [p.im_start_id] + self._encode("user") + p.newline_ids
        patches, grids, hashes = [], [], []
        for image in images:
            camera = image["camera"]
            if self.image_mode == "no_tele" and camera == "front_tele":
                continue
            label, size = CAMERAS[camera]
            if self.image_mode == "blank":
                pil, digest = Image.new("RGB", size, (127, 127, 127)), "blank"
            else:
                pil, digest = load_image(image, self.image_root)
            hashes.append(digest)
            patch, (rows, cols) = p._patchify(CameraFrame(pil, target_size=size), self.config.current_image_pixels)
            count = rows * cols // p.merge_size ** 2
            prompt += self._encode(f"{label} relative_time={float(image.get('relative_time', 0)):g}s\n")
            prompt += [p.vision_start_id] + [p.image_token_id] * count + [p.vision_end_id]
            patches.append(patch)
            grids.append((1, rows, cols))
        student_text = validate_student(record["student_obs"])
        prompt += self._encode("\nStudent road, ego and perception observation:\n" + student_text)
        state_position = len(prompt) - 1
        prompt += self._encode("\nChoose the next executable driving behavior. Candidate conditions are part of the behavior.\n")
        positions = []
        for index, candidate in enumerate(candidates):
            body = {key: candidate.get(key) for key in ("action_type", "target_id", "speed_profile", "description")}
            prompt += self._encode(f"Candidate {index + 1}: " + json.dumps(body, ensure_ascii=False, separators=(",", ":")))
            positions.append(len(prompt) - 1)
            prompt += self._encode("\n")
        prompt += [p.im_end_id] + p.newline_ids + [p.im_start_id] + self._encode("assistant") + p.newline_ids
        prompt += self._encode("Decision:")
        decision_position = len(prompt) - 1
        if len(prompt) > self.max_tokens:
            raise ValueError(f"Prompt {len(prompt)} exceeds max_tokens={self.max_tokens}; no silent truncation")
        input_ids = torch.tensor([prompt], dtype=torch.long)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "mm_token_type_ids": (input_ids == p.image_token_id).long()}
        if patches:
            inputs.update(pixel_values=torch.cat(patches), image_grid_thw=torch.tensor(grids, dtype=torch.long))
        targets = None
        target_map = (record.get("targets") or {}).get("meta_action", {}).get("probabilities")
        if target_map is not None:
            if set(target_map) != set(ids):
                raise ValueError("Targets must map exactly to the presented candidate IDs")
            targets = torch.tensor([float(target_map[k]) for k in ids], dtype=torch.float32)
            if abs(float(targets.sum()) - 1) > 1e-5 or (targets < 0).any():
                raise ValueError("Invalid teacher distribution")
        metadata = {"compiler_version": COMPILER_VERSION, "tokens": len(prompt), "image_sha256": hashes, "image_mode": self.image_mode,
                    "request_candidate_ids": list(layout.request_ids), "record_id": record.get("record_id"),
                    "prompt_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()}
        return CompiledDecision(inputs, decision_position, positions, state_position, ids, [c["action_type"] for c in candidates], targets, metadata)
