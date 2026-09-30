
"""
LD-GCN style implementation for remote sensing multi-label classification.

Compatible with:
    output = model(images)
    output["logits"]

Core:
- ResNet101 visual feature extractor
- label-driven graph reasoning
- label relation propagation
- image-label fusion
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


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.fc(adj @ x)


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if torchvision is None:
            raise ImportError("torchvision is required")

        try:
            net = torchvision.models.resnet101(
                weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
        except Exception:
            net = torchvision.models.resnet101(pretrained=pretrained)

        self.net = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, 1)


class LDGCN(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = Backbone(pretrained_backbone)

        self.label_embedding = nn.Parameter(
            torch.randn(num_classes, 512)
        )
        nn.init.normal_(self.label_embedding, std=0.02)

        self.gcn1 = GraphConv(512, 1024)
        self.gcn2 = GraphConv(1024, 2048)

        self.label_relation = nn.Parameter(
            torch.randn(num_classes, num_classes)
        )

        nn.init.normal_(self.label_relation, std=0.02)

        self.fusion = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(
            2048,
            num_classes
        )

    def get_graph(self):
        adj = torch.sigmoid(self.label_relation)

        degree = adj.sum(
            dim=1,
            keepdim=True
        ).clamp(min=1e-6)

        return adj / degree

    def forward(self, images):

        visual = self.backbone(images)

        adj = self.get_graph()

        label = F.relu(
            self.gcn1(
                self.label_embedding,
                adj
            )
        )

        label = self.gcn2(
            label,
            adj
        )

        label = F.normalize(
            label,
            dim=-1
        )

        relation = torch.matmul(
            F.normalize(visual, dim=-1),
            label.t()
        )

        label_feature = torch.matmul(
            relation,
            label
        )

        fused = torch.cat(
            [
                visual,
                label_feature
            ],
            dim=-1
        )

        fused = self.fusion(fused)

        logits = self.classifier(fused)

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


def build_ld_gcn(
    num_classes,
    pretrained_backbone=True,
    **kwargs
):
    return LDGCN(
        num_classes,
        pretrained_backbone
    )
