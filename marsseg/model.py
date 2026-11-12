"""Segmentation networks: SegFormer (MiT-B2 encoder) or DeepLabv3+ (ResNet-50), both ImageNet-pretrained,
with a dropout layer in front of the classifier so the same weights give Monte-Carlo dropout samples."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

ARCHS = {
    "segformer-b2": lambda w: smp.Segformer("mit_b2", encoder_weights=w, in_channels=3, classes=4),
    "deeplabv3p-r50": lambda w: smp.DeepLabV3Plus("resnet50", encoder_weights=w, in_channels=3, classes=4),
}


class MCDropout2d(nn.Dropout2d):
    """Dropout2d that can be switched on at inference independently of model.eval()."""

    mc = False

    def forward(self, x):
        return nn.functional.dropout2d(x, self.p, self.training or self.mc)


def build(arch: str = "segformer-b2", pretrained: bool = True, p_drop: float = 0.1) -> nn.Module:
    m = ARCHS[arch]("imagenet" if pretrained else None)
    head = m.segmentation_head
    head[0] = nn.Sequential(MCDropout2d(p_drop), head[0])
    m.arch = arch
    return m


def set_mc(model: nn.Module, on: bool) -> None:
    for mod in model.modules():
        if isinstance(mod, MCDropout2d):
            mod.mc = on


def load(path, device="cuda") -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["arch"], pretrained=False, p_drop=ck.get("p_drop", 0.1))
    m.load_state_dict(ck["state"])
    m.temperature = ck.get("temperature", 1.0)
    return m.to(device).eval()
