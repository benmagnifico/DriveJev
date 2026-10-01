"""Load the Qwen-Drive-1.0 vision-language model (without its planning expert and LM head)."""
import gc

import torch


def load_backbone(model_path, device="cuda:0", dtype=torch.bfloat16, attn_implementation="eager", lora_path=None):
    """Return the Qwen-Drive VLM body (`vlm.model`) in eval mode with frozen weights.

    `lora_path` optionally merges a PEFT LoRA adapter into the language layers before use
    (released DriveJev weights are already merged, so this is normally None).
    """
    from qwen_drive import QwenDriveForPlanning
    container = QwenDriveForPlanning.from_pretrained(model_path, dtype=dtype, device_map={"": "cpu"},
                                                     attn_implementation=attn_implementation)
    backbone = container.vlm.model
    backbone.config.text_config.use_cache = False
    del container
    gc.collect()
    if lora_path:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, lora_path).merge_and_unload()
    backbone.requires_grad_(False)
    return backbone.to(device=device, dtype=dtype).eval()
