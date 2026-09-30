"""
ADD-GCN-style dynamic label graph adapted to the unified comparison Trainer.

Fixes in this version:
1. dynamic adjacency is [B, C, C] and is no longer a rank-1 outer product;
2. graph convolution is batched (no Python loop over images);
3. final logits have unrestricted BCEWithLogitsLoss-compatible range;
4. output remains {"logits": [B, C]} and parameter_groups is unchanged.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import torchvision
from torchvision.models import ResNet101_Weights


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x:   [B, C, Din] (or [C, Din])
        adj: [B, C, C]   (or [C, C])
        """
        support = torch.matmul(x, self.weight)
        return torch.matmul(adj, support) + self.bias


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        model = torchvision.models.resnet101(
            weights=(
                ResNet101_Weights.IMAGENET1K_V2
                if pretrained else None
            )
        )
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(self.features(x), 1)


class DynamicGraphGenerator(nn.Module):
    """
    Build an image-conditioned full label-label adjacency matrix.

    A global image feature produces a feature-dimension gate. The same label
    embeddings are reweighted by that gate before pairwise label affinities are
    computed, so each image can change the complete CxC relation structure.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.image_gate = nn.Linear(feature_dim, hidden_dim)

    def forward(self, feature: Tensor, label_embedding: Tensor) -> Tensor:
        # feature: [B, 2048], label_embedding: [C, 256]
        gate = torch.sigmoid(self.image_gate(feature))  # [B, 256]

        labels = label_embedding.unsqueeze(0)  # [1, C, 256]
        conditioned = labels * (1.0 + gate.unsqueeze(1))

        scores = torch.matmul(
            conditioned,
            conditioned.transpose(1, 2),
        ) / math.sqrt(self.hidden_dim)

        adj = torch.softmax(scores, dim=-1)

        # Preserve self information and re-normalize rows.
        c = label_embedding.size(0)
        eye = torch.eye(c, device=feature.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj


class ADDGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        self.backbone = ResNet101Backbone(pretrained_backbone)

        self.label_embedding = nn.Parameter(
            torch.empty(self.num_classes, 256)
        )
        nn.init.normal_(self.label_embedding, std=0.02)

        self.graph_generator = DynamicGraphGenerator(2048, 256)
        self.gcn1 = GraphConvolution(256, 512)
        self.gcn2 = GraphConvolution(512, 2048)

        self.visual_projection = nn.Linear(2048, 2048)
        self.visual_norm = nn.LayerNorm(2048)
        self.node_norm = nn.LayerNorm(2048)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, images: Tensor):
        visual = self.backbone(images)  # [B, 2048]
        adj = self.graph_generator(visual, self.label_embedding)  # [B,C,C]

        b = visual.size(0)
        nodes = self.label_embedding.unsqueeze(0).expand(b, -1, -1)
        nodes = F.relu(self.gcn1(nodes, adj), inplace=False)
        nodes = self.gcn2(nodes, adj)  # [B, C, 2048]

        visual = self.visual_norm(self.visual_projection(visual))
        nodes = self.node_norm(nodes)

        # Scaled dot-product classifier: unlike cosine similarity, logits are
        # not clipped to [-1, 1] and remain appropriate for BCEWithLogitsLoss.
        logits = torch.einsum("bd,bcd->bc", visual, nodes)
        logits = logits / math.sqrt(visual.size(-1))
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


def build_add_gcn(
    num_classes: int,
    pretrained_backbone: bool = True,
    **kwargs,
):
    return ADDGCN(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        **kwargs,
    )
