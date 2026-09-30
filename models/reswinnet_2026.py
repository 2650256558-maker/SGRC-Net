"""ReSwinNet (2026) - runnable paper-aligned reimplementation.

Reference
---------
Ningthoujam et al., "Optimizing multi-label image annotation: a hybrid
CNN-Transformer deep learning approach", Scientific Reports, 2026.

This file implements the architecture described in the paper's method section:
  ResNet-50 stem/layer1/layer2 multi-scale features
  -> 1x1 projections 64->32, 256->48, 512->96
  -> upsample/concatenate/compress 176->96
  -> bypass Swin patch embedding and feed 96-D tokens directly into Swin-T
  -> hierarchical Swin stages -> GAP -> LayerNorm -> multi-label classifier.

The paper also studies frequency-based label pruning / sampling. Those dataset-
specific preprocessing steps are intentionally NOT applied here because the thesis
comparison must evaluate every method on the same complete label space and splits.
Therefore this is the architectural ReSwinNet comparison under the common protocol,
not an official-author code release.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torchvision.models import (
        ResNet50_Weights,
        Swin_T_Weights,
        resnet50,
        swin_t,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("torchvision with resnet50 and swin_t is required") from exc


def _safe_resnet50(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as exc:
            print(f"[ReSwinNet2026][Warning] ResNet-50 pretrained weights unavailable: {exc}")
    return resnet50(weights=None)


def _safe_swin_t(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            return swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        except Exception as exc:
            print(f"[ReSwinNet2026][Warning] Swin-T pretrained weights unavailable: {exc}")
    return swin_t(weights=None)


class ReSwinNet2026(nn.Module):
    """ResNet-Swin token-projection fusion network for multi-label classification."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        # Keep the full modules so pretrained parameter accounting remains close
        # to the paper; forward uses the stages explicitly stated in the method.
        self.resnet = _safe_resnet50(pretrained=pretrained)
        self.swin = _safe_swin_t(pretrained=pretrained)

        # Paper: F1 64->32, F2 256->48, F3 512->96, concat 176->96.
        self.proj_f1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.proj_f2 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.proj_f3 = nn.Conv2d(512, 96, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 48 + 96, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )

        # The Swin-T patch embedding is bypassed. features[1:] starts with the
        # first 96-D Swin stage and accepts BHWC tensors.
        self.swin_stages = self.swin.features[1:]
        self.norm = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, self.num_classes)

    def _resnet_multiscale(self, x: Tensor):
        r = self.resnet
        x = r.conv1(x)
        x = r.bn1(x)
        x = r.relu(x)
        f1 = r.maxpool(x)          # [B, 64, H/4, W/4]
        f2 = r.layer1(f1)          # [B,256,H/4,W/4]
        f3 = r.layer2(f2)          # [B,512,H/8,W/8]
        return f1, f2, f3

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        f1, f2, f3 = self._resnet_multiscale(x)
        p1 = self.proj_f1(f1)
        p2 = self.proj_f2(f2)
        p3 = self.proj_f3(f3)
        p3 = F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([p1, p2, p3], dim=1))  # B,96,H/4,W/4

        # Torchvision Swin blocks use BHWC after patch embedding.
        tokens = fused.permute(0, 2, 3, 1).contiguous()
        tokens = self.swin_stages(tokens)                  # B,h,w,768
        pooled = tokens.mean(dim=(1, 2))
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}

    def parameter_groups(self, backbone_lr: float, head_lr: float) -> List[dict]:
        # ResNet and Swin are pretrained backbones; new projection/fusion/head
        # use the normal comparison-model LR.
        backbone_params = list(self.resnet.parameters()) + list(self.swin_stages.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": head_params, "lr": float(head_lr)},
        ]


def build_model(num_classes: int, pretrained: bool = True) -> ReSwinNet2026:
    return ReSwinNet2026(num_classes=num_classes, pretrained=pretrained)
