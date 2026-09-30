
"""
CAGRN style implementation for remote sensing multi-label classification.

Compatible interface:
    output = model(images)
    output["logits"]

Core idea:
- CNN visual representation
- Cross-attention driven adaptive graph reasoning
- Label relation refinement
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import ResNet101_Weights
except Exception:
    torchvision = None
    ResNet101_Weights = None


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = torchvision.models.resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.net = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, 1)


class GraphReasoning(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.query = nn.Linear(dim, dim // 4)
        self.key = nn.Parameter(torch.randn(num_classes, dim // 4))
        self.value = nn.Parameter(torch.randn(num_classes, dim))

        nn.init.normal_(self.key, std=0.02)
        nn.init.normal_(self.value, std=0.02)

    def forward(self, feature):
        q = self.query(feature)
        att = torch.softmax(
            torch.matmul(q, self.key.t()),
            dim=-1
        )

        graph_feature = torch.matmul(
            att,
            self.value
        )

        return graph_feature


class CAGRN(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        **kwargs
    ):
        super().__init__()

        self.backbone = Backbone(pretrained_backbone)

        self.graph = GraphReasoning(
            2048,
            num_classes
        )

        self.fusion = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(
            2048,
            num_classes
        )

    def forward(self, images):

        visual = self.backbone(images)

        relation = self.graph(
            visual
        )

        fused = torch.cat(
            [
                visual,
                relation
            ],
            dim=-1
        )

        fused = self.fusion(fused)

        logits = self.classifier(
            fused
        )

        return {
            "logits": logits
        }

    def parameter_groups(
        self,
        backbone_lr,
        head_lr
    ):
        return [
            {
                "params": self.backbone.parameters(),
                "lr": float(backbone_lr)
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("backbone.")
                ],
                "lr": float(head_lr)
            }
        ]


def build_cagrn(
    num_classes,
    pretrained_backbone=True,
    **kwargs
):
    return CAGRN(
        num_classes,
        pretrained_backbone
    )
