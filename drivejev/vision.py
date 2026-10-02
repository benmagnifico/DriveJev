"""Image patching for the vision-language backbone.

Resizes a camera frame onto the backbone's patch grid and flattens it into patches, with the
vision-token ids the prompt needs. The patch layout and resizing rules follow the Qwen-Drive-1.0
processor (Apache License 2.0, https://huggingface.co/Qwen/Qwen-Drive-1.0-4B), so the backbone
sees exactly the pixels it was trained on.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Snap a resolution to a multiple of `factor` inside a pixel budget."""
    bar_h, bar_w = round(height / factor) * factor, round(width / factor) * factor
    if bar_h * bar_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        bar_h, bar_w = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif bar_h * bar_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        bar_h, bar_w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return bar_h, bar_w


def local_folder(model_path):
    """A local backbone folder, downloading the files the backbone needs when `model_path` is a Hub repo id."""
    if Path(model_path).is_dir():
        return Path(model_path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(model_path), allow_patterns=["*.json", "*.jinja", "*.txt", "model*.safetensors*"]))


class ImagePatcher:
    def __init__(self, model_path, tokenizer):
        config = json.loads((local_folder(model_path) / "config.json").read_text())
        vlm = config["vlm_config"]
        self.patch_size = config["image_patch_size"]
        self.merge_size = config["image_spatial_merge_size"]
        self.temporal_patch_size = config["image_temporal_patch_size"]
        self.factor = self.patch_size * self.merge_size
        self.min_pixels = 4 * self.factor ** 2
        self.grid_pixel_limit = 12800 * self.factor ** 2
        self.image_token_id = vlm["image_token_id"]
        self.vision_start_id = vlm["vision_start_token_id"]
        self.vision_end_id = vlm["vision_end_token_id"]
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    def patchify(self, image, target_size):
        """`image`: RGB PIL image; `target_size`: (width, height). Returns (patches, (rows, cols))."""
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF
        width, height = target_size
        image = TF.resize(image.convert("RGB"), [height, width], interpolation=InterpolationMode.BICUBIC)
        width, height = image.size
        grid_height, grid_width = smart_resize(height, width, self.factor, self.min_pixels, self.grid_pixel_limit)
        image = TF.resize(image, [grid_height, grid_width], interpolation=InterpolationMode.BICUBIC)
        pixels = TF.pil_to_tensor(image).float().div_(255.0).sub_(0.5).div_(0.5)
        rows, cols, merge, patch, temporal = (grid_height // self.patch_size, grid_width // self.patch_size, self.merge_size,
                                              self.patch_size, self.temporal_patch_size)
        pixels = pixels.unsqueeze(1).expand(-1, temporal, -1, -1)
        patches = pixels.reshape(pixels.shape[0], 1, temporal, rows // merge, merge, patch, cols // merge, merge, patch)
        return patches.permute(1, 3, 6, 4, 7, 0, 2, 5, 8).reshape(rows * cols, -1), (rows, cols)
