
"""
SIGNA style implementation.

Compatible interface:
    output = model(images)
    output["logits"]

Components:
- ResNet101 visual encoder
- label graph reasoning
- semantic interaction
- channel attention fusion
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
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.linear(adj @ x)


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        net = torchvision.models.resnet101(
            weights=(
                ResNet101_Weights.IMAGENET1K_V2
                if pretrained else None
            )
        )

        self.net = nn.Sequential(
            *list(net.children())[:-1]
        )

    def forward(self,x):
        x=self.net(x)
        return torch.flatten(x,1)


class SIGNA(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        **kwargs
    ):
        super().__init__()

        self.num_classes=num_classes

        self.backbone=Backbone(
            pretrained_backbone
        )

        self.label_embedding=nn.Parameter(
            torch.randn(num_classes,512)
        )

        self.gcn=GraphConv(
            512,
            2048
        )

        self.channel_attention=nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,2048),
            nn.Sigmoid()
        )

        self.classifier=nn.Linear(
            2048,
            num_classes
        )

    def forward(self,images):

        visual=self.backbone(images)

        adj=torch.eye(
            self.num_classes,
            device=images.device
        )

        label=self.gcn(
            self.label_embedding,
            adj
        )

        label=F.normalize(
            label,
            dim=-1
        )

        semantic=torch.matmul(
            F.normalize(visual,dim=-1),
            label.t()
        )

        att=self.channel_attention(
            visual
        )

        fused=visual*att

        logits=self.classifier(
            fused
        )

        logits=logits+semantic

        return {
            "logits":logits
        }

    def parameter_groups(
        self,
        backbone_lr,
        head_lr
    ):
        return [
            {
                "params":self.backbone.parameters(),
                "lr":float(backbone_lr)
            },
            {
                "params":[
                    p for n,p in self.named_parameters()
                    if not n.startswith("backbone.")
                ],
                "lr":float(head_lr)
            }
        ]


def build_signa(
    num_classes,
    pretrained_backbone=True,
    **kwargs
):
    return SIGNA(
        num_classes,
        pretrained_backbone
    )
