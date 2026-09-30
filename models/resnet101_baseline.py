"""Locked standalone ResNet-101 baseline for ablation experiment A.

Design goal
-----------
This file is deliberately independent from ``glr_drcf_net.py``. Experiment A
must remain a fixed reference even when the GLR/CMRDE/local-refinement/DRCF
model is modified later.

Architecture
------------
Image -> torchvision ResNet-101 -> global average pooling -> Linear(C)

No graph branch, FAVOR+, CMRDE, local refinement, LS-SCNA or DRCF is imported
or instantiated here.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

try:
    import torchvision
    from torchvision.models import ResNet101_Weights
except Exception:  # pragma: no cover
    torchvision = None
    ResNet101_Weights = None


def _strip_common_prefixes(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Remove common wrappers from externally saved ResNet checkpoints."""
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "backbone."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _build_torchvision_resnet101(
    pretrained: bool = True,
    local_weights: str = "",
) -> nn.Module:
    """Construct an ordinary torchvision ResNet-101, independent of GLR code."""
    if torchvision is None:
        raise ImportError("torchvision is required for ResNet101Baseline.")

    try:
        weights = (
            ResNet101_Weights.IMAGENET1K_V2
            if pretrained and not local_weights
            else None
        )
        model = torchvision.models.resnet101(weights=weights)
    except Exception:
        # Compatibility with older torchvision versions.
        model = torchvision.models.resnet101(
            pretrained=bool(pretrained and not local_weights)
        )

    if local_weights:
        checkpoint = torch.load(local_weights, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Unsupported ResNet101 baseline checkpoint format: {local_weights}"
            )

        state_dict = _strip_common_prefixes(checkpoint)
        # Local ImageNet checkpoints normally contain fc.*. If a checkpoint
        # comes from another classifier, ignore an incompatible fc safely;
        # experiment A always creates its own task-specific classifier below.
        if "fc.weight" in state_dict and state_dict["fc.weight"].shape[0] != 1000:
            state_dict.pop("fc.weight", None)
            state_dict.pop("fc.bias", None)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Baseline-A] Loaded local ResNet101 weights: {local_weights}")
        if missing:
            print(f"[Baseline-A] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Baseline-A] Unexpected keys: {len(unexpected)}")

    return model


class ResNet101Baseline(nn.Module):
    """Fixed pure-CNN reference used only by ablation experiment A."""

    baseline_id = "torchvision_resnet101_gap_linear_v1"

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        backbone_weights: str = "",
    ) -> None:
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive")

        model = _build_torchvision_resnet101(
            pretrained=pretrained_backbone,
            local_weights=backbone_weights,
        )

        # Keep the conventional ResNet-101 feature extractor exactly as
        # torchvision defines it. Only the 1000-way ImageNet FC is replaced by
        # the dataset-specific multi-label classifier.
        in_features = int(model.fc.in_features)  # 2048
        model.fc = nn.Linear(in_features, int(num_classes))
        self.model = model

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        # Reproduce torchvision forward explicitly so the pooled 2048-D feature
        # can still be exposed for diagnostics without depending on GLR code.
        x = self.model.conv1(images)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        logits = self.model.fc(feature)
        return {
            "logits": logits,
            "cnn_feature": feature,
        }

    def parameter_groups(self, backbone_lr: float, head_lr: float):
        """Use the same optimizer interface as the full model."""
        backbone_params = []
        for name, parameter in self.model.named_parameters():
            if not name.startswith("fc."):
                backbone_params.append(parameter)
        return [
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": self.model.fc.parameters(), "lr": float(head_lr)},
        ]


def build_resnet101_baseline(
    num_classes: int,
    pretrained_backbone: bool = True,
    backbone_weights: str = "",
) -> ResNet101Baseline:
    return ResNet101Baseline(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        backbone_weights=backbone_weights,
    )
