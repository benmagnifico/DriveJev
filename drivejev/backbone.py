"""Load the vision-language backbone body (vision encoder + language model, no LM head)."""
import json

import torch


def load_backbone(model_path, device="cuda:0", dtype=torch.bfloat16, attn_implementation="eager", lora_path=None):
    """Return the VLM body in eval mode with frozen weights, loaded with plain `transformers`.

    `model_path` is the released `backbone/` folder: its config holds the VLM configuration under
    `vlm_config` and its weights are stored under the `vlm.model.` prefix.
    `lora_path` optionally merges a PEFT LoRA adapter into the language layers before use.
    """
    from transformers import AutoConfig, AutoModel
    from .vision import local_folder
    model_path = local_folder(model_path)
    from transformers.utils import logging
    vlm_config = AutoConfig.for_model(**json.loads((model_path / "config.json").read_text())["vlm_config"])
    verbosity = logging.get_verbosity()
    logging.set_verbosity_error()   # the folder's config names the full driving model; only its VLM body is loaded
    try:
        backbone = AutoModel.from_pretrained(model_path, config=vlm_config, dtype=dtype, device_map={"": "cpu"},
                                             attn_implementation=attn_implementation, key_mapping={r"^vlm\.model\.": ""})
    finally:
        logging.set_verbosity(verbosity)
    backbone.config.text_config.use_cache = False
    if lora_path:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, lora_path).merge_and_unload()
    backbone.requires_grad_(False)
    return backbone.to(device=device, dtype=dtype).eval()
