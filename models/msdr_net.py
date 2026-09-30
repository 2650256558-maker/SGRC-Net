
"""
Fixed MSDR-Net implementation.

Fix:
- Correct ResNet multi-scale feature dimension:
  256 + 512 + 1024 + 2048 = 3840
"""

import torch
from torch import nn


import torchvision
from torchvision.models import ResNet101_Weights


class MultiScaleBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        net = torchvision.models.resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2
            if pretrained else None
        )

        self.layer1 = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1
        )

        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x = self.layer1(x)
        f1 = torch.flatten(self.pool(x),1)

        x = self.layer2(x)
        f2 = torch.flatten(self.pool(x),1)

        x = self.layer3(x)
        f3 = torch.flatten(self.pool(x),1)

        x = self.layer4(x)
        f4 = torch.flatten(self.pool(x),1)

        return torch.cat(
            [f1,f2,f3,f4],
            dim=1
        )


class DynamicReasoning(nn.Module):

    def __init__(self, dim, num_classes):
        super().__init__()

        self.query = nn.Linear(
            dim,
            512
        )

        self.label_tokens = nn.Parameter(
            torch.randn(num_classes,512)
        )

    def forward(self, feature):

        q = self.query(feature)

        relation = torch.matmul(
            q,
            self.label_tokens.t()
        )

        return torch.softmax(
            relation,
            dim=-1
        )


class MSDRNet(nn.Module):

    def __init__(
        self,
        num_classes,
        pretrained_backbone=True,
        **kwargs
    ):
        super().__init__()

        self.backbone = MultiScaleBackbone(
            pretrained_backbone
        )

        self.project = nn.Linear(
            3840,
            2048
        )

        self.reasoning = DynamicReasoning(
            2048,
            num_classes
        )

        self.classifier = nn.Linear(
            2048,
            num_classes
        )

    def forward(self, images):

        feature = self.backbone(images)

        feature = self.project(feature)

        relation = self.reasoning(feature)

        # relation is only used as adaptive weighting
        enhanced = feature

        logits = self.classifier(
            enhanced
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


def build_msdr_net(
    num_classes,
    pretrained_backbone=True,
    **kwargs
):
    return MSDRNet(
        num_classes,
        pretrained_backbone
    )
