"""
ML-GCN implementation adapted to the unified comparison Trainer.

Key compatibility points:
- output is a dict containing ``logits`` with shape [B, num_classes]
- exposes ``parameter_groups(backbone_lr, head_lr)``
- explicitly requests one-time label-graph initialization from the Trainer
- label graph is built ONLY from the training split targets

The classifier follows the ML-GCN idea: GCN-refined label representations act
as class-specific classifier weights for the CNN visual feature.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import ResNet101_Weights
except Exception:
    torchvision = None
    ResNet101_Weights = None


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        return adj @ x @ self.weight + self.bias


class LabelGraphBuilder:
    """Build a stable symmetric label co-occurrence graph from train targets."""

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)

    def build(self, targets: Tensor) -> Tensor:
        if targets.ndim != 2 or targets.shape[1] != self.num_classes:
            raise ValueError(
                f"targets must have shape [N,{self.num_classes}], "
                f"got {tuple(targets.shape)}"
            )

        targets = targets.float()
        co = targets.T @ targets
        freq = torch.diag(co).clamp(min=1.0)

        # P(j|i), then symmetrize so graph propagation is numerically stable.
        conditional = co / freq.unsqueeze(1)
        adj = 0.5 * (conditional + conditional.T)
        adj.fill_diagonal_(1.0)

        # Symmetric degree normalization: D^{-1/2} A D^{-1/2}.
        degree = adj.sum(dim=1).clamp(min=1e-6)
        inv_sqrt = degree.rsqrt()
        adj = inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
        return adj


class ResNet101Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if torchvision is None:
            raise ImportError("torchvision is required for ML-GCN.")

        try:
            model = torchvision.models.resnet101(
                weights=(
                    ResNet101_Weights.IMAGENET1K_V2
                    if pretrained else None
                )
            )
        except Exception:
            model = torchvision.models.resnet101(pretrained=pretrained)

        self.features = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = 2048

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class MLGCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        # Trainer checks only this explicit flag. Other models are untouched.
        self.requires_label_graph_init = True

        self.backbone = ResNet101Backbone(pretrained=pretrained_backbone)

        self.label_embedding = nn.Parameter(
            torch.empty(self.num_classes, 256)
        )
        nn.init.normal_(self.label_embedding, std=0.02)

        self.graph_builder = LabelGraphBuilder(self.num_classes)
        self.gcn1 = GraphConvolution(256, 512)
        self.gcn2 = GraphConvolution(512, 2048)

        # LayerNorm + scaled dot-product keeps initialization numerically
        # well-behaved without constraining logits to the cosine [-1, 1] range.
        self.visual_norm = nn.LayerNorm(2048)
        self.label_norm = nn.LayerNorm(2048)
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))

        self.register_buffer(
            "label_adj",
            torch.eye(self.num_classes, dtype=torch.float32),
        )
        self._graph_ready = False

    @torch.no_grad()
    def update_label_graph(self, targets: Tensor) -> None:
        adj = self.graph_builder.build(targets.detach().cpu())
        self.label_adj.copy_(adj.to(self.label_adj.device))
        self._graph_ready = True

    def forward(self, images: Tensor):
        visual = self.backbone(images)  # [B, 2048]

        if self._graph_ready:
            adj = self.label_adj
        else:
            # This fallback keeps FLOPs/FPS profiling in main_comparison.py
            # functional before Trainer performs train-label initialization.
            adj = torch.eye(
                self.num_classes,
                device=images.device,
                dtype=visual.dtype,
            )

        labels = F.relu(self.gcn1(self.label_embedding, adj), inplace=False)
        labels = self.gcn2(labels, adj)  # [C, 2048]

        # Layer-normalized scaled dot product: stable at initialization while
        # remaining free to produce logits far beyond the cosine [-1, 1] range.
        visual = self.visual_norm(visual)
        labels = self.label_norm(labels)
        logits = (visual @ labels.T) / (visual.size(-1) ** 0.5)
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


def build_ml_gcn(
    num_classes: int,
    pretrained_backbone: bool = True,
    **kwargs,
):
    return MLGCN(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        **kwargs,
    )
