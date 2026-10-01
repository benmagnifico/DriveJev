"""DriveJev decision heads over frozen Qwen-Drive hidden states.

pointer      : linear FP32 q(decision) . k(candidate) / sqrt(d)              (1.31 M params)
pointer_mlp  : q = MLP([decision; state]), k = MLP(candidate), LayerNorm inputs  (4.2 M params)
"""
import math

import torch
from torch import nn


class PointerHead(nn.Module):
    kind = "pointer"

    def __init__(self, hidden_size=2560, projection_dim=256, **unused):
        super().__init__()
        self.q = nn.Linear(hidden_size, projection_dim)
        self.k = nn.Linear(hidden_size, projection_dim)
        self.scale = 1 / math.sqrt(projection_dim)
        self.config = {"kind": self.kind, "hidden_size": hidden_size, "projection_dim": projection_dim}

    def forward(self, decision, candidates, state=None, mask=None):
        with torch.autocast(device_type=decision.device.type, enabled=False):
            logits = (self.k(candidates.float()) * self.q(decision.float()).unsqueeze(-2)).sum(-1) * self.scale
            return logits if mask is None else logits.masked_fill(~mask.bool(), -torch.inf)


class PointerMLPHead(nn.Module):
    kind = "pointer_mlp"

    def __init__(self, hidden_size=2560, projection_dim=256, width=512, dropout=0.1, **unused):
        super().__init__()
        self.norm_d = nn.LayerNorm(hidden_size)
        self.norm_s = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)
        self.q = nn.Sequential(nn.Linear(2 * hidden_size, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, projection_dim))
        self.k = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, projection_dim))
        self.scale = 1 / math.sqrt(projection_dim)
        self.config = {"kind": self.kind, "hidden_size": hidden_size, "projection_dim": projection_dim, "width": width, "dropout": dropout}

    def forward(self, decision, candidates, state=None, mask=None):
        with torch.autocast(device_type=decision.device.type, enabled=False):
            if state is None:
                raise ValueError("pointer_mlp needs the student-state hidden vector")
            q = self.q(torch.cat([self.norm_d(decision.float()), self.norm_s(state.float())], -1))
            logits = (self.k(self.norm_c(candidates.float())) * q.unsqueeze(-2)).sum(-1) * self.scale
            return logits if mask is None else logits.masked_fill(~mask.bool(), -torch.inf)


HEADS = {cls.kind: cls for cls in (PointerHead, PointerMLPHead)}


def build_head(config):
    return HEADS[config["kind"]](**{k: v for k, v in config.items() if k != "kind"})
