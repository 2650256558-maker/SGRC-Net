"""
S-MAT-style semantic token model adapted to the unified comparison Trainer.

Fixes in this version:
- the image feature is inserted as a visual token into the Transformer, so
  semantic label tokens are conditioned on the current image;
- final logits use scaled dot products rather than cosine values restricted to
  [-1, 1];
- output remains {"logits": [B, num_classes]}.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

import torchvision
from torchvision.models import ResNet101_Weights


class Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = torchvision.models.resnet101(
            weights=(
                ResNet101_Weights.IMAGENET1K_V2
                if pretrained else None
            )
        )
        self.net = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(self.net(x), 1)


class SMAT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = 2048

        self.backbone = Backbone(pretrained_backbone)

        self.label_tokens = nn.Parameter(
            torch.empty(self.num_classes, self.feature_dim)
        )
        nn.init.normal_(self.label_tokens, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=4096,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # Each image-conditioned semantic label token is mapped to one logit.
        # A shared head preserves the semantic-token formulation and produces
        # a BCEWithLogitsLoss-friendly scale at initialization.
        self.classifier = nn.Linear(self.feature_dim, 1)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, images: Tensor):
        visual = self.backbone(images)  # [B, 2048]
        b = visual.size(0)

        labels = self.label_tokens.unsqueeze(0).expand(b, -1, -1)

        # Crucial compatibility fix: label tokens and the current image token
        # are processed together, so semantic relations are image-conditioned.
        sequence = torch.cat([visual.unsqueeze(1), labels], dim=1)
        sequence = self.transformer(sequence)

        # The label-token outputs have already interacted with the image token.
        label_tokens = sequence[:, 1:, :]      # [B, C, 2048]
        logits = self.classifier(label_tokens).squeeze(-1)
        logits = logits + self.class_bias

        return {"logits": logits}

    def parameter_groups(self, backbone_lr: float, head_lr: float):
        return [
            {
                "params": self.backbone.parameters(),
                "lr": float(backbone_lr),
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("backbone.")
                ],
                "lr": float(head_lr),
            },
        ]


def build_smat(
    num_classes: int,
    pretrained_backbone: bool = True,
    **kwargs,
):
    return SMAT(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        **kwargs,
    )
