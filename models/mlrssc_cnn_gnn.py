
"""
MLRSSC-CNN-GNN re-implementation.

Compatible with:
    output = model(images)
    output["logits"]

Designed for the unified comparison framework:
AAAAA/
├── main_comparison.py
└── models/
    └── mlrssc_cnn_gnn.py

Core components:
- ResNet101 visual branch
- spatial-aware region relation approximation
- graph reasoning branch
- CNN-GNN feature fusion
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import ResNet101_Weights
except Exception:
    torchvision = None
    ResNet101_Weights = None


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.fc(torch.matmul(adj, x))


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if torchvision is None:
            raise ImportError("torchvision required")

        try:
            net = torchvision.models.resnet101(
                weights=ResNet101_Weights.IMAGENET1K_V2
                if pretrained else None
            )
        except Exception:
            net = torchvision.models.resnet101(
                pretrained=pretrained
            )

        self.features = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


class SpatialRelationModule(nn.Module):
    """
    Approximate spatial relation reasoning.
    Generates image-specific relation weights.
    """

    def __init__(self, dim, num_classes):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Parameter(
            torch.randn(num_classes, dim)
        )
        nn.init.normal_(self.key, std=0.02)

    def forward(self, feature):
        q = self.query(feature)
        relation = torch.matmul(
            q,
            self.key.t()
        )
        return torch.softmax(relation, dim=-1)


class MLRSSCCNNGNN(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = ResNet101Backbone(
            pretrained_backbone
        )

        self.label_nodes = nn.Parameter(
            torch.randn(num_classes, 512)
        )

        self.graph = GraphConv(
            512,
            2048
        )

        self.relation = SpatialRelationModule(
            2048,
            num_classes
        )

        self.fusion = nn.Linear(
            4096,
            2048
        )

        self.classifier = nn.Linear(
            2048,
            num_classes
        )

    def forward(self, images):

        visual = self.backbone(images)

        relation = self.relation(
            visual
        )

        label_feature = self.graph(
            self.label_nodes,
            torch.eye(
                self.num_classes,
                device=images.device
            )
        )

        label_feature = F.normalize(
            label_feature,
            dim=-1
        )

        graph_feature = torch.matmul(
            relation.unsqueeze(1),
            label_feature.unsqueeze(0)
        ).squeeze(1)

        fused = torch.cat(
            [
                visual,
                graph_feature
            ],
            dim=-1
        )

        fused = F.relu(
            self.fusion(fused)
        )

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
                    p for n,p in self.named_parameters()
                    if not n.startswith("backbone.")
                ],
                "lr": float(head_lr)
            }
        ]


def build_mlrssc_cnn_gnn(
    num_classes,
    pretrained_backbone=True,
    **kwargs
):
    return MLRSSCCNNGNN(
        num_classes,
        pretrained_backbone
    )
