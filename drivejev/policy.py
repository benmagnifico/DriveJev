"""DrivingPolicy: frozen Qwen-Drive-1.0 encoder + DriveJev decision head(s).

    policy = DrivingPolicy.from_pretrained("path/to/DriveJev-4B", device="cuda:0")
    out = policy.predict(record)   # record = {student_obs, candidates, images:[front -0.5, front 0, front_tele 0]}
    out["candidate_id"], out["probabilities"]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .backbone import load_backbone
from .compiler import DriveCompiler
from .heads import build_head


class Encoder:
    """Runs the VLM and returns the hidden states at the decision, candidate and state tokens."""

    def __init__(self, backbone_path, device="cuda:0", image_mode="normal", lora_path=None):
        self.backbone = load_backbone(backbone_path, device=device, lora_path=lora_path)
        self.compiler = DriveCompiler(backbone_path, image_mode=image_mode)
        self.device = next(self.backbone.parameters()).device
        self.pad_id = self.compiler.tokenizer.pad_token_id or 0

    @torch.no_grad()
    def encode(self, compiled_list):
        """Right-padded batch: causal attention and left-to-right linear attention leave real tokens unchanged."""
        rows = [c.to(self.device) for c in compiled_list]
        positions = []
        for row in rows:
            inp = row.model_inputs
            pos, _ = self.backbone.get_rope_index(input_ids=inp["input_ids"], mm_token_type_ids=inp["mm_token_type_ids"],
                                                  image_grid_thw=inp.get("image_grid_thw"), attention_mask=inp["attention_mask"])
            positions.append(pos)
        length = max(r.model_inputs["input_ids"].shape[1] for r in rows)

        def pad(t, value):
            return torch.nn.functional.pad(t, (0, length - t.shape[-1]), value=value)

        batch = {"input_ids": torch.cat([pad(r.model_inputs["input_ids"], self.pad_id) for r in rows]),
                 "attention_mask": torch.cat([pad(r.model_inputs["attention_mask"], 0) for r in rows]),
                 "mm_token_type_ids": torch.cat([pad(r.model_inputs["mm_token_type_ids"], 0) for r in rows]),
                 "position_ids": torch.cat([pad(p, 0) for p in positions], dim=1)}
        if any("pixel_values" in r.model_inputs for r in rows):
            batch["pixel_values"] = torch.cat([r.model_inputs["pixel_values"] for r in rows if "pixel_values" in r.model_inputs])
            batch["image_grid_thw"] = torch.cat([r.model_inputs["image_grid_thw"] for r in rows if "image_grid_thw" in r.model_inputs])
        hidden = self.backbone(**batch, use_cache=False, output_hidden_states=False, return_dict=True).last_hidden_state
        return [(hidden[i][r.decision_position], hidden[i][r.candidate_positions], hidden[i][r.state_position]) for i, r in enumerate(rows)]


def load_head(path, device="cpu"):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        config = json.loads((path.parent / "drivejev_config.json").read_text())
        head = build_head(config["head"])
        head.load_state_dict(load_file(str(path)), strict=True)
        head.temperature = float(config.get("temperature", 1.0))
        return head.to(device).eval(), config
    saved = torch.load(path, map_location="cpu", weights_only=False)
    head = build_head(saved["head_config"])
    head.load_state_dict(saved["state_dict"], strict=True)
    head.temperature = float(saved.get("temperature", 1.0))
    return head.to(device).eval(), saved


GO = {"keep_route_cruise", "proceed_route", "route_turn"}


def choose(candidate_ids, action_types, probabilities, current_action=None, rule="argmax"):
    """'argmax', or 'group': decide move vs. slow/stop by summed probability first, then take the most likely
    behaviour of that group (behaviours with identical outcomes share probability mass, which otherwise biases
    plain argmax toward behaviours that have no twin)."""
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    if rule != "group":
        return candidate_ids[best]
    is_go = [a in GO or (a == "continue_current" and current_action == "route_turn") for a in action_types]
    go = sum(p for p, g in zip(probabilities, is_go) if g)
    group = [i for i, g in enumerate(is_go) if g == (go > 1 - go)]
    return candidate_ids[max(group, key=probabilities.__getitem__)] if group else candidate_ids[best]


class DrivingPolicy:
    def __init__(self, encoder: Encoder, heads: dict, default_arm=None, decision_rule="argmax"):
        self.encoder, self.heads, self.decision_rule = encoder, heads, decision_rule
        self.default_arm = default_arm or next(iter(heads))

    @classmethod
    def from_pretrained(cls, path, device="cuda:0", image_mode="normal", base_model=None):
        """`path`: a released DriveJev folder (drivejev_config.json + head.safetensors + backbone/), or a
        Hugging Face repo id. `base_model` overrides where the backbone is read from."""
        root = Path(path)
        if not root.exists():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(str(path)))
        config = json.loads((root / "drivejev_config.json").read_text())
        backbone = config.get("backbone", "backbone")
        backbone = base_model or (str(root / backbone) if (root / backbone).exists() else backbone)
        encoder = Encoder(str(backbone), device=device, image_mode=image_mode)
        head, _ = load_head(root / config.get("head_file", "head.safetensors"), encoder.device)
        return cls(encoder, {"default": head}, decision_rule=config.get("decision_rule", "argmax"))

    @torch.no_grad()
    def predict(self, record, arm=None):
        started = time.perf_counter()
        compiled = self.encoder.compiler.compile_record(record)
        compiled_at = time.perf_counter()
        (decision, candidates, state), = self.encoder.encode([compiled])
        if self.encoder.device.type == "cuda":
            torch.cuda.synchronize()
        encoded_at = time.perf_counter()
        head = self.heads[arm or self.default_arm]
        logits = head(decision.unsqueeze(0), candidates.unsqueeze(0), state.unsqueeze(0))[0]
        probabilities = (logits / getattr(head, "temperature", 1.0)).softmax(-1).cpu().tolist()
        done = time.perf_counter()
        mapped = dict(zip(compiled.candidate_ids, probabilities))
        best = choose(compiled.candidate_ids, compiled.action_types, probabilities,
                      (record.get("student_obs") or {}).get("maneuver", {}).get("action_type"), self.decision_rule)
        return {"candidate_id": best, "probabilities": {k: mapped[k] for k in compiled.metadata["request_candidate_ids"]},
                "observation_id": record.get("observation_id"), "confidence": mapped[best], "tokens": compiled.metadata["tokens"],
                "timing_ms": {"preprocess": (compiled_at - started) * 1000, "backbone": (encoded_at - compiled_at) * 1000,
                              "head": (done - encoded_at) * 1000, "total": (done - started) * 1000}}
