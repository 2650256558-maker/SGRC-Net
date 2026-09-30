"""SPiT (2026) - runnable paper-aligned reimplementation.

Reference
---------
Li et al., "Superpixel-informed transformer for multi-label image
classification", Neurocomputing 695 (2026), 133995.

Public descriptions specify a decompose-then-aggregate design:
  differentiable superpixel decomposition -> region Transformer encoder ->
  learnable class-query Transformer decoder -> gated evidence fusion.

The publisher page does not expose official source code or every low-level
hyperparameter. This implementation therefore reproduces the published mechanism
in a clean form compatible with this project's common Trainer. It is NOT claimed
to be the authors' official implementation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet50_Weights, resnet50
except Exception as exc:  # pragma: no cover
    raise ImportError("torchvision with resnet50 is required") from exc


def _safe_resnet50(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as exc:
            print(f"[SPiT2026][Warning] ResNet-50 pretrained weights unavailable: {exc}")
    return resnet50(weights=None)


class DifferentiableSuperpixelTokenizer(nn.Module):
    """Learn soft spatial assignments and aggregate coherent region tokens.

    The assignment maps are generated jointly from visual features and normalized
    2-D coordinates, making the decomposition trainable end-to-end while retaining
    an explicit spatial prior.
    """

    def __init__(self, in_dim: int, embed_dim: int = 256, num_regions: int = 64) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_regions = int(num_regions)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.assignment = nn.Sequential(
            nn.Conv2d(embed_dim + 2, embed_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_regions, 1),
        )
        self.token_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _coord_grid(b: int, h: int, w: int, device, dtype) -> Tensor:
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0)
        return grid.expand(b, -1, -1, -1)

    def forward(self, fmap: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.feature_proj(fmap)
        b, d, h, w = feat.shape
        coords = self._coord_grid(b, h, w, feat.device, feat.dtype)
        assign_logits = self.assignment(torch.cat([feat, coords], dim=1))

        # For each region, normalize evidence over spatial positions. This makes
        # each token an adaptive weighted superpixel-like aggregation.
        weights = assign_logits.flatten(2).softmax(dim=-1)   # B,K,HW
        pixels = feat.flatten(2).transpose(1, 2)             # B,HW,D
        tokens = torch.bmm(weights, pixels)                  # B,K,D
        tokens = self.token_norm(tokens)
        return tokens, weights.view(b, self.num_regions, h, w)


class SPiT2026(nn.Module):
    """Superpixel-informed class-query Transformer for multi-label classification."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        embed_dim: int = 256,
        num_regions: int = 64,
        num_heads: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = _safe_resnet50(pretrained=pretrained)

        # Use ResNet layer3 spatial features; the final layer4/classifier are not
        # used for prediction, keeping region evidence sufficiently fine-grained.
        self.tokenizer = DifferentiableSuperpixelTokenizer(
            in_dim=1024, embed_dim=embed_dim, num_regions=num_regions
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.class_queries = nn.Parameter(torch.randn(self.num_classes, embed_dim) * 0.02)

        # Gated fusion combines class-specific decoded evidence with image-level
        # region context, matching the paper's unified aggregation/fusion idea.
        self.global_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def _backbone_layer3(self, x: Tensor) -> Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        return x

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        fmap = self._backbone_layer3(x)
        regions, assignments = self.tokenizer(fmap)
        memory = self.encoder(regions)

        b = x.size(0)
        queries = self.class_queries.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(queries, memory)

        global_ctx = self.global_proj(memory.mean(dim=1)).unsqueeze(1).expand_as(decoded)
        gate = self.gate(torch.cat([decoded, global_ctx], dim=-1))
        fused = gate * decoded + (1.0 - gate) * global_ctx
        fused = self.out_norm(fused)
        logits = self.classifier(fused).squeeze(-1)

        # Only logits are consumed by the shared trainer; region maps are kept
        # detached for optional future visualization without affecting loss code.
        return {"logits": logits}

    def parameter_groups(self, backbone_lr: float, head_lr: float) -> List[dict]:
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": head_params, "lr": float(head_lr)},
        ]


def build_model(num_classes: int, pretrained: bool = True) -> SPiT2026:
    return SPiT2026(num_classes=num_classes, pretrained=pretrained)
