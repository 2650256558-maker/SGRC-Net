"""
Chapter-5 Complementarity-Preserving ASRC (CP-ASRC) + Lite-SADRF Variant
==================================================================
Global-guided Local Refinement + Unidirectional Graph Relation Correction
for multi-label remote-sensing image classification.

Third-version-derived semantic-residual pipeline:
1) standard ResNet-101 remains unchanged and provides Layer3 + Layer4 features
2) Layer3 (14x14 at 224 input) forms fine-grained graph nodes
3) standard FAVOR+ Performer performs global relation modeling
4) B5 relation-confidence residual calibration filters uncertain FAVOR+ updates
5) B1 label-specific global readout extracts class-aware global evidence
6) U_rel / U_cls / U_local are fused by CMRDE
7) Layer4 high-level semantics estimates node semantic relevance A_sem
8) final routing demand R = R_CMRDE * A_sem selects Top-K nodes
9) R still guides the local semantic edge gate, preserving a differentiable CMRDE path
10) C1 decouples routing demand from local residual magnitude with one learnable scalar alpha_local
11) C2 replaces fixed Top-25% routing with image-adaptive budgeted sparse routing
12) per-image threshold = mean(R) + std(R); selected count is clamped by the original refine-ratio budget
13) C3 changes local residual injection to zero-start signed ReZero-style scaling alpha_local=tanh(s_local)
14) semantic-edge-gated Max-Relative GraphConv + LS-SCNA refine active selected nodes
15) Layer4 GAP preserves an independent CNN baseline prediction path
16) DRCF produces a relation-correction representation rather than replacing CNN
17) final logits = CNN logits + qG*(1-qC)*graph correction logits

Compared with the original third version, the CNN is still a standard ResNet-101;
only feature usage, graph routing, and CNN-GNN fusion are changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import ResNet101_Weights
except Exception:  # pragma: no cover
    torchvision = None
    ResNet101_Weights = None


# -----------------------------------------------------------------------------
# Backbone utilities
# -----------------------------------------------------------------------------


def _strip_state_dict_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Strip common wrappers such as ``module.`` and ``backbone.`` from keys."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def build_resnet101_backbone(
    pretrained: bool = True,
    local_weights: str = "",
) -> nn.Module:
    """Build ResNet-101 and optionally load a local ImageNet-style checkpoint."""
    if torchvision is None:
        raise ImportError("torchvision is required to build the ResNet-101 backbone.")

    # Modern torchvision API, with a fallback for older installations.
    try:
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained and not local_weights else None
        model = torchvision.models.resnet101(weights=weights)
    except Exception:
        model = torchvision.models.resnet101(pretrained=pretrained and not local_weights)

    if local_weights:
        checkpoint = torch.load(local_weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported backbone checkpoint format: {local_weights}")
        checkpoint = _strip_state_dict_prefix(checkpoint)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f"[Backbone] Loaded local weights: {local_weights}")
        if missing:
            print(f"[Backbone] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Backbone] Unexpected keys: {len(unexpected)}")

    return model


class ResNet101FeatureExtractor(nn.Module):
    """
    Standard ResNet-101 feature extractor exposing Layer3 and Layer4.

    The CNN architecture itself is NOT modified. The model only reuses two
    existing stages for different roles:
      - Layer3: fine-grained graph nodes (1024 channels, ~14x14 at 224 input)
      - Layer4: high-level CNN semantic anchor (2048 channels, ~7x7)

    ``forward(x)`` still returns Layer4 only for backward compatibility with
    the pure ResNet101 baseline used by the ablation runner.
    """

    layer3_channels = 1024
    out_channels = 2048

    def __init__(self, pretrained: bool = True, local_weights: str = "") -> None:
        super().__init__()
        model = build_resnet101_backbone(pretrained=pretrained, local_weights=local_weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward_hierarchy(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return Layer2/Layer3/Layer4 without changing the pretrained trunk."""
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer2, layer3, layer4

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        _, layer3, layer4 = self.forward_hierarchy(x)
        return layer3, layer4

    def forward(self, x: Tensor) -> Tensor:
        _, layer4 = self.forward_features(x)
        return layer4



# -----------------------------------------------------------------------------
# Chapter-5 V14: gradient-isolated Stage-3 semantic Lite-SADRF sidecar
# -----------------------------------------------------------------------------


class _LiteDWConvNorm(nn.Sequential):
    """Depthwise Conv + GroupNorm branch for batch-stable lightweight DRCE."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        effective = dilation * (kernel_size - 1) + 1
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=effective // 2,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8 if channels % 8 == 0 else 4, channels),
        )


class LiteDRCESemanticStage3(nn.Module):
    """Very small DRCE sidecar on the *pure* ResNet-101 Layer3 feature.

    Earlier variants used lower-level context.  That kept the GNN safe, but pooled Layer2
    evidence was too low-level for multi-label correction: the learned label
    residual was active yet did not improve the CNN ranking.  V14 therefore
    reads Layer3, where semantics are substantially richer in ResNet-101.

    Importantly, the sidecar reads ``stage3.detach()`` and never writes back to
    Layer3.  SGLR-GNN still receives the original Layer3 tensor unchanged.

    The context extractor stays lightweight by doing all large/dilated spatial
    operations after a 1024 -> hidden projection and never expanding a spatial
    feature back to 1024 channels.
    """

    def __init__(self, in_channels: int = 1024, hidden_channels: int = 48) -> None:
        super().__init__()
        h = int(hidden_channels)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, h, 1, bias=False),
            nn.GroupNorm(8 if h % 8 == 0 else 4, h),
            nn.GELU(),
        )
        # DRCE core: one large-kernel depthwise branch + two dilated branches.
        # At inference this remains tiny because it operates on only h channels.
        self.large = _LiteDWConvNorm(h, 5, dilation=1)
        self.dilated1 = _LiteDWConvNorm(h, 3, dilation=1)
        self.dilated2 = _LiteDWConvNorm(h, 3, dilation=2)
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(h, h, 1, bias=False),
            nn.GroupNorm(8 if h % 8 == 0 else 4, h),
        )
        se_hidden = max(h // 4, 12)
        self.recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(h, se_hidden, 1),
            nn.GELU(),
            nn.Conv2d(se_hidden, h, 1),
            nn.Sigmoid(),
        )
        # Small non-zero start so the context path learns from batch 1.
        self.context_raw = nn.Parameter(torch.tensor(0.15))

    def forward(self, stage3: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        local = self.reduce(stage3.detach())
        delta = self.large(local) + self.dilated1(local) + self.dilated2(local)
        delta = self.mix(delta)
        delta = delta * self.recalibrate(delta)
        strength = torch.tanh(self.context_raw)
        context = local + strength * delta
        return local, context, strength


class LiteSemanticAdaptiveFusion(nn.Module):
    """SADRF-style scene-adaptive local/global coordination in reduced space.

    The router is channel-wise but operates on only ``hidden`` channels.  It
    retains the Chapter-3 adaptive local/global weighting idea without the
    expensive full-resolution difference-enhancement branch.
    """

    def __init__(self, channels: int = 48, hidden: int = 32, local_prior: float = 0.70) -> None:
        super().__init__()
        if not 0.0 < local_prior < 1.0:
            raise ValueError("local_prior must be in (0,1)")
        c = int(channels)
        self.channels = c
        self.router = nn.Sequential(
            nn.Conv2d(c * 4, int(hidden), 1),
            nn.GELU(),
            nn.Conv2d(int(hidden), c * 2, 1),
        )
        final = self.router[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            final.bias[:c].fill_(math.log(local_prior))
            final.bias[c:].fill_(math.log(1.0 - local_prior))

    def forward(self, local: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        desc = torch.cat(
            [
                F.adaptive_avg_pool2d(local, 1),
                F.adaptive_max_pool2d(local, 1),
                F.adaptive_avg_pool2d(context, 1),
                F.adaptive_max_pool2d(context, 1),
            ],
            dim=1,
        )
        logits = self.router(desc)
        b = logits.size(0)
        weights = torch.softmax(
            logits.view(b, 2, self.channels, 1, 1), dim=1
        )
        fused = weights[:, 0] * local + weights[:, 1] * context
        balance = weights.mean(dim=2).flatten(1)
        return fused, balance


class LiteDualTargetVisualResidualHead(nn.Module):
    """Semantic-conditioned dual-target residual head.

    V14 used the *same* label residual for both the pure CNN prediction and the
    already-fused SGRC prediction. Those two targets are not identical: one
    correction may improve the CNN while being sub-optimal after GNN/ASRC fusion.

    V15 keeps one shared SADRF visual descriptor but learns two tiny class heads:
      - ``delta_cnn``   : improves the pure CNN branch;
      - ``delta_final`` : improves the complete SGRC prediction.

    Both heads read only detached SGRC features, so the original SGRC optimization
    path remains untouched. This is still a genuine visual enhancement module,
    not checkpoint calibration or post-hoc interpolation.
    """

    def __init__(
        self,
        context_channels: int,
        layer4_channels: int,
        num_classes: int,
        hidden: int = 96,
        delta_max: float = 0.30,
    ) -> None:
        super().__init__()
        c = int(context_channels)
        h = int(hidden)
        self.delta_max_final = float(delta_max)
        self.delta_max_cnn = float(delta_max) * 0.85
        if self.delta_max_final <= 0:
            raise ValueError("delta_max must be positive")

        self.semantic_proj = nn.Sequential(
            nn.Linear(int(layer4_channels), c, bias=False),
            nn.LayerNorm(c),
            nn.GELU(),
        )

        # Multi-statistic descriptor: mean, max, std, local-context difference,
        # and the stable Layer4 semantic anchor.  The extra std term adds almost
        # no compute but helps retain distribution information lost by GAP.
        descriptor_dim = c * 5
        self.norm = nn.LayerNorm(descriptor_dim)
        self.trunk = nn.Sequential(
            nn.Linear(descriptor_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h, bias=False),
            nn.GELU(),
        )
        self.cnn_delta_head = nn.Linear(h, int(num_classes))
        self.final_delta_head = nn.Linear(h, int(num_classes))
        nn.init.normal_(self.cnn_delta_head.weight, mean=0.0, std=1.5e-3)
        nn.init.zeros_(self.cnn_delta_head.bias)
        nn.init.normal_(self.final_delta_head.weight, mean=0.0, std=1.5e-3)
        nn.init.zeros_(self.final_delta_head.bias)

    def forward(
        self,
        local: Tensor,
        context: Tensor,
        fused: Tensor,
        layer4_feature: Tensor,
        pure_cnn_logits: Tensor,
        sgrc_logits: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        avg_fused = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        max_fused = F.adaptive_max_pool2d(fused, 1).flatten(1)
        std_fused = fused.float().flatten(2).std(dim=-1, unbiased=False)
        diff = F.adaptive_avg_pool2d(torch.abs(context - local), 1).flatten(1)
        semantic = self.semantic_proj(layer4_feature.detach().float())
        desc = self.norm(
            torch.cat([avg_fused, max_fused, std_fused, diff, semantic], dim=-1).float()
        )
        hidden = self.trunk(desc)

        cnn_delta = self.delta_max_cnn * torch.tanh(self.cnn_delta_head(hidden))
        final_delta = self.delta_max_final * torch.tanh(self.final_delta_head(hidden))

        enhanced_cnn = pure_cnn_logits.detach().float() + cnn_delta
        enhanced_final = sgrc_logits.detach().float() + final_delta

        with torch.no_grad():
            cnn_prob_shift = torch.abs(
                torch.sigmoid(enhanced_cnn)
                - torch.sigmoid(pure_cnn_logits.detach().float())
            ).mean(dim=-1)
            final_prob_shift = torch.abs(
                torch.sigmoid(enhanced_final)
                - torch.sigmoid(sgrc_logits.detach().float())
            ).mean(dim=-1)

        aux = {
            "lite_cnn_delta_abs_mean": cnn_delta.detach().abs().mean(dim=-1),
            "lite_cnn_delta_max_abs": cnn_delta.detach().abs().amax(dim=-1),
            "lite_final_delta_abs_mean": final_delta.detach().abs().mean(dim=-1),
            "lite_final_delta_max_abs": final_delta.detach().abs().amax(dim=-1),
            # Backward-compatible aliases used by existing summary code.
            "lite_visual_delta_abs_mean": final_delta.detach().abs().mean(dim=-1),
            "lite_visual_delta_max_abs": final_delta.detach().abs().amax(dim=-1),
            "lite_residual_risk": torch.ones_like(final_delta[:, 0]).detach(),
            "lite_cnn_probability_shift": cnn_prob_shift.detach(),
            "lite_final_probability_shift": final_prob_shift.detach(),
        }
        return (
            enhanced_cnn.to(pure_cnn_logits.dtype),
            cnn_delta,
            final_delta,
            aux,
        )


class LiteSADRFSemanticResidualAdapter(nn.Module):
    """V15 sidecar: Stage3 Lite-DRCE + adaptive L/G + semantic residual.

    The sidecar reads detached SGRC features only.  It never writes into Layer3,
    Layer4, CMRDE, SGLR-GNN or ASRC.  Training is gradient-isolated by the V14
    trainer, so the original SGRC optimization remains unchanged while the
    visual branch learns a complementary class-wise correction.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 48,
        delta_max: float = 0.35,
    ) -> None:
        super().__init__()
        h = int(hidden_channels)
        self.drce = LiteDRCESemanticStage3(1024, h)
        self.fusion = LiteSemanticAdaptiveFusion(
            h, hidden=max(h // 2, 24), local_prior=0.70
        )
        self.residual_head = LiteDualTargetVisualResidualHead(
            context_channels=h,
            layer4_channels=2048,
            num_classes=num_classes,
            hidden=max(96, h * 2),
            delta_max=delta_max,
        )

    def forward(
        self,
        stage3: Tensor,
        layer4_feature: Tensor,
        pure_cnn_logits: Tensor,
        sgrc_logits: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        local, context, strength = self.drce(stage3)
        fused, balance = self.fusion(local, context)
        enhanced_cnn, cnn_delta, final_delta, aux = self.residual_head(
            local=local,
            context=context,
            fused=fused,
            layer4_feature=layer4_feature,
            pure_cnn_logits=pure_cnn_logits,
            sgrc_logits=sgrc_logits,
        )
        aux.update(
            {
                "lite_context_strength": strength.detach(),
                "lite_local_weight": balance[:, 0].detach(),
                "lite_global_weight": balance[:, 1].detach(),
            }
        )
        return enhanced_cnn, cnn_delta, final_delta, aux


# -----------------------------------------------------------------------------
# Position encoding + Performer-style linear attention
# -----------------------------------------------------------------------------


def build_2d_sincos_position_encoding(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return [1, H*W, D] deterministic 2-D sinusoidal positional encoding."""
    if dim % 4 != 0:
        # Pad to the next multiple of 4 and truncate afterwards.
        padded_dim = int(math.ceil(dim / 4.0) * 4)
    else:
        padded_dim = dim

    quarter = padded_dim // 4
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter, 1)))

    pe = torch.cat(
        [
            torch.sin(xx * omega),
            torch.cos(xx * omega),
            torch.sin(yy * omega),
            torch.cos(yy * omega),
        ],
        dim=1,
    )
    pe = pe[:, :dim].unsqueeze(0).to(dtype=dtype)
    return pe



class FAVORPlusPerformerAttention(nn.Module):
    """
    Standard FAVOR+ Performer attention using positive orthogonal random features.

    The module approximates softmax attention without constructing an N x N
    attention matrix. Random projections are registered as buffers so checkpoints
    preserve the exact feature map used by the experiment.

    It additionally derives a node-wise Global Relation Uncertainty U_rel from the
    second moment (concentration) of the approximate relation distribution:
        H_i = sum_j a_ij^2
    computed from FAVOR+ sufficient statistics, again without reconstructing the
    N x N matrix. Diffuse global relations -> high uncertainty.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_features: int = 64,
        orthogonal_scaling: int = 0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if orthogonal_scaling not in (0, 1):
            raise ValueError("orthogonal_scaling must be 0 or 1")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.num_features = int(num_features)
        self.orthogonal_scaling = int(orthogonal_scaling)
        self.eps = float(eps)

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        projection = self._gaussian_orthogonal_random_matrix(
            self.num_features,
            self.head_dim,
            scaling=self.orthogonal_scaling,
        )
        self.register_buffer("projection_matrix", projection, persistent=True)

    @staticmethod
    def _orthogonal_matrix_chunk(cols: int) -> Tensor:
        # QR is computed in float32 for stable orthogonalization.
        unstructured = torch.randn(cols, cols, dtype=torch.float32)
        q, _ = torch.linalg.qr(unstructured, mode="reduced")
        return q.t().contiguous()

    @classmethod
    def _gaussian_orthogonal_random_matrix(
        cls,
        nb_rows: int,
        nb_columns: int,
        scaling: int = 0,
    ) -> Tensor:
        blocks = []
        full_blocks = nb_rows // nb_columns
        for _ in range(full_blocks):
            blocks.append(cls._orthogonal_matrix_chunk(nb_columns))
        remaining = nb_rows - full_blocks * nb_columns
        if remaining > 0:
            blocks.append(cls._orthogonal_matrix_chunk(nb_columns)[:remaining])
        matrix = torch.cat(blocks, dim=0)

        if scaling == 0:
            multiplier = torch.randn(nb_rows, nb_columns, dtype=torch.float32).norm(dim=1)
        else:
            multiplier = math.sqrt(float(nb_columns)) * torch.ones(nb_rows, dtype=torch.float32)
        return matrix * multiplier.unsqueeze(-1)

    def redraw_projection_matrix(self) -> None:
        """Optional manual redraw; normally keep fixed for reproducible experiments."""
        with torch.no_grad():
            projection = self._gaussian_orthogonal_random_matrix(
                self.num_features,
                self.head_dim,
                scaling=self.orthogonal_scaling,
            ).to(device=self.projection_matrix.device)
            self.projection_matrix.copy_(projection)

    def _softmax_kernel(self, data: Tensor, is_query: bool) -> Tensor:
        """
        Positive FAVOR+ random feature map for the softmax kernel.

        Args:
            data: [B,H,N,Dh]
        Returns:
            features: [B,H,N,M], float32 for numerical stability.
        """
        data_f = data.float()
        projection = self.projection_matrix.to(device=data.device, dtype=torch.float32)
        data_normalizer = float(self.head_dim) ** -0.25
        ratio = float(self.num_features) ** -0.5

        data_scaled = data_f * data_normalizer
        projected = torch.einsum("bhnd,md->bhnm", data_scaled, projection)
        diag = (data_scaled.square().sum(dim=-1, keepdim=True)) * 0.5

        # Multiplicative stabilizers cancel after attention normalization.
        if is_query:
            stabilizer = projected.max(dim=-1, keepdim=True).values.detach()
        else:
            stabilizer = projected.amax(dim=(-2, -1), keepdim=True).detach()

        features = ratio * (
            torch.exp(projected - diag - stabilizer) + self.eps
        )
        return features

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Numerically stable FAVOR+ forward; attention statistics are FP32."""
        b, n, d = x.shape
        # Performer is small compared with ResNet; keep this sensitive block in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = x.float()
            x_norm = self.norm(residual)
            qkv = self.qkv(x_norm).reshape(b, n, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            q_phi = self._softmax_kernel(q, is_query=True)
            k_phi = self._softmax_kernel(k, is_query=False)

            kv = torch.einsum("bhnm,bhnd->bhmd", k_phi, v)
            k_sum = k_phi.sum(dim=2)
            denom = torch.einsum("bhnm,bhm->bhn", q_phi, k_sum)
            denom_safe = denom.clamp_min(self.eps)
            out = torch.einsum("bhnm,bhmd->bhnd", q_phi, kv) / denom_safe.unsqueeze(-1)
            out = self.attn_dropout(out).transpose(1, 2).reshape(b, n, d)
            out = self.proj_dropout(self.out_proj(out))
            global_nodes = residual + out

            # U_rel is a routing/evidence statistic, not an optimization target.
            # Detaching prevents high-order rational gradients from destabilizing Q/K.
            with torch.no_grad():
                q_u = q_phi.detach(); k_u = k_phi.detach()
                denom_u = torch.einsum("bhnm,bhm->bhn", q_u, k_u.sum(dim=2))
                k_second = torch.einsum("bhnm,bhnp->bhmp", k_u, k_u)
                second_num = torch.einsum("bhnm,bhmp,bhnp->bhn", q_u, k_second, q_u)
                # Do NOT clamp denom^2 to 1e-6: that badly biases concentration when denom<1e-3.
                second_moment = second_num / (denom_u.square() + 1e-12)
                if n > 1:
                    uniform = 1.0 / float(n)
                    concentration = ((second_moment - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
                else:
                    concentration = torch.ones_like(second_moment)
                relation_concentration = concentration.mean(dim=1)
                relation_uncertainty = (1.0 - relation_concentration).clamp(0.0, 1.0)

        diagnostics = {"linear_attention_denominator": denom_safe.detach(), "relation_concentration": relation_concentration.detach()}
        return global_nodes, relation_uncertainty, diagnostics


# -----------------------------------------------------------------------------
# B5: Relation-Confidence Residual Calibration
# -----------------------------------------------------------------------------


class RelationConfidenceResidualCalibration(nn.Module):
    """Calibrate the FAVOR+ residual update using relation uncertainty.

    FAVOR+ produces a relation update ``delta = G_raw - X`` for every spatial
    node. B5 keeps the attention mechanism itself unchanged and only calibrates
    how strongly that update is injected:

        G = X + gate * delta
        gate = 1 - s * U_rel

    where ``U_rel`` is the node-wise relation uncertainty already estimated by
    FAVOR+, and ``s`` is a single learnable calibration strength.

    ``s = 0.5 * tanh(raw_strength)`` is initialized to exactly zero. Therefore
    B5 is *exactly identical to B1 at initialization* and adds no random-number
    consumption that could change the initialization of the existing modules.

    Positive ``s`` suppresses uncertain relation updates (the intended
    confidence-calibration behaviour); negative ``s`` lets optimization test
    whether the uncertainty signal should instead be used in the opposite
    direction. The bounded range keeps the residual gate in [0.5, 1.5].

    This is a GLOBAL-relation calibration only. It does not fuse CNN/GNN logits
    and does not perform local refinement, so the roles of CMRDE/local modules
    and DRCF remain unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        # Scalar only; deterministic zero initialization preserves B1 exactly.
        self.raw_strength = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        base_nodes: Tensor,
        raw_global_nodes: Tensor,
        relation_uncertainty: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if base_nodes.shape != raw_global_nodes.shape:
            raise ValueError(
                "base_nodes and raw_global_nodes must have the same shape, "
                f"got {tuple(base_nodes.shape)} vs {tuple(raw_global_nodes.shape)}"
            )
        if relation_uncertainty.shape != base_nodes.shape[:2]:
            raise ValueError(
                "relation_uncertainty must be [B,N], "
                f"got {tuple(relation_uncertainty.shape)} for nodes {tuple(base_nodes.shape)}"
            )

        # Keep this very small calibration block in FP32.
        with torch.autocast(device_type=base_nodes.device.type, enabled=False):
            x = base_nodes.float()
            g_raw = raw_global_nodes.float()
            u_rel = relation_uncertainty.float().clamp(0.0, 1.0)

            # Bounded signed strength in [-0.5, 0.5], exactly 0 at init.
            strength = 0.5 * torch.tanh(self.raw_strength.float())
            gate = 1.0 - strength * u_rel             # [B,N]
            delta = g_raw - x                          # [B,N,D]
            calibrated = x + gate.unsqueeze(-1) * delta

        diagnostics = {
            "relation_calibration_strength": strength.detach(),
            "relation_residual_gate_mean": gate.mean().detach(),
            "relation_residual_gate_min": gate.amin().detach(),
            "relation_residual_gate_max": gate.amax().detach(),
        }
        return calibrated, diagnostics


# -----------------------------------------------------------------------------
# CMRDE: Context-aware Multi-Evidence Refinement Demand Estimator
# -----------------------------------------------------------------------------


class CMRDE(nn.Module):
    """
    Fuse three uncertainty evidences with dynamic reliability weighting and
    pairwise evidence interactions to produce a per-node refinement score R.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.reliability = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        # [u_rel, u_cls, u_local, weighted_u, pair12, pair13, pair23]
        self.score = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, u_rel: Tensor, u_cls: Tensor, u_local: Tensor) -> Tuple[Tensor, Tensor]:
        # Evidence routing is tiny; keep it FP32 even under AMP.
        with torch.autocast(device_type=u_rel.device.type, enabled=False):
            evidence = torch.stack([u_rel.float(), u_cls.float(), u_local.float()], dim=-1).clamp(0.0, 1.0)
            rel_weights = torch.softmax(self.reliability(evidence), dim=-1)
            weighted = (rel_weights * evidence).sum(dim=-1, keepdim=True)
            interactions = torch.stack([evidence[...,0]*evidence[...,1], evidence[...,0]*evidence[...,2], evidence[...,1]*evidence[...,2]], dim=-1)
            r = torch.sigmoid(self.score(torch.cat([evidence, weighted, interactions], dim=-1))).squeeze(-1)
        return r, rel_weights


# -----------------------------------------------------------------------------
# Sparse global-guided local refinement
# -----------------------------------------------------------------------------


def _build_local_neighbor_index(
    height: int,
    width: int,
    kernel_size: int,
    device: torch.device,
) -> Tensor:
    """Build [N, K] clipped local spatial-neighbor indices."""
    radius = kernel_size // 2
    indices = []
    for yy in range(height):
        for xx in range(width):
            neigh = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = min(max(yy + dy, 0), height - 1)
                    nx = min(max(xx + dx, 0), width - 1)
                    neigh.append(ny * width + nx)
            indices.append(neigh)
    return torch.tensor(indices, device=device, dtype=torch.long)



class LabelSemanticGuidedSCNA(nn.Module):
    """
    D4: Soft-Floor Reliability-Weighted Label Semantic Separation (SFRSS).

    D3 correctly avoided aggressive correction for extremely ambiguous nodes,
    but its reliable-ambiguity weight becomes exactly zero when Top-1 and Top-2
    are nearly tied:
        w_D3 = 4 * m_i * (1 - m_i)

    That can be too conservative for recall-sensitive hard nodes. D4 therefore
    keeps D3's bell-shaped reliability term and adds a SMALL low-margin recovery
    term that vanishes for clear nodes:

        m_i      = (p1 - p2) / (p1 + p2 + eps)
        w_bell   = 4 * m_i * (1 - m_i)
        w_floor  = 0.25 * (1 - m_i)^2
        w_i      = clamp(w_bell + w_floor, 0, 1)

    Behaviour:
      - m_i -> 1 (very clear):     w_i -> 0, no semantic correction
      - m_i -> 0 (extremely tied): w_i -> 0.25, allow only weak correction
      - m_i ~ 0.5:                 w_i -> 1, strongest correction

    The coefficient 0.25 is shared globally, not class-specific and not
    dataset-specific. No new learnable module is introduced.

    The discriminative direction is unchanged:
        d_i = normalize(proto_top1 - proto_top2) * sqrt(D)

    Final semantic residual:
        g_i = sigmoid(f(z_i, d_i, w_i, R_i))
        alpha_sem = tanh(s_sem), s_sem initialized to 0
        z'_i = z_i + alpha_sem * g_i * w_i * d_i

    D1's identity-preserving / zero-start design is retained.
    R remains gate context only.
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 0.2,
        dropout: float = 0.1,  # API compatibility; intentionally unused in D2
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.dim = int(dim)
        self.temperature = float(temperature)
        self.eps = float(eps)

        # Supervised classifier weights are mapped into a semantic prototype space.
        self.label_encoder = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
        )

        hidden = max(dim // 2, 64)

        # [selected node, discriminative direction, ambiguity, refinement score]
        self.gate = nn.Sequential(
            nn.Linear(dim * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Exact identity over C3 at initialization.
        self.semantic_residual_raw = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        selected_nodes: Tensor,
        global_nodes: Tensor,
        label_prototypes: Tensor,
        selected_indices: Tensor,
        refinement_score: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, n, d = global_nodes.shape
        c = label_prototypes.size(0)

        with torch.autocast(device_type=global_nodes.device.type, enabled=False):
            selected_f = selected_nodes.float()
            proto_in = label_prototypes.float()

            # Label prototypes come from the supervised node classifier.
            proto = F.normalize(
                self.label_encoder(proto_in),
                dim=-1,
                eps=1e-6,
            )  # [C,D]

            selected_unit = F.normalize(
                selected_f,
                dim=-1,
                eps=1e-6,
            )  # [B,K,D]

            # Node-to-label semantic assignment.
            logits = (
                torch.einsum("bkd,cd->bkc", selected_unit, proto)
                / self.temperature
            )
            assignment = torch.softmax(logits, dim=-1)

            # Entropy is kept only as a diagnostic for compatibility.
            if c > 1:
                a_safe = assignment.clamp_min(self.eps)
                entropy_selected = (
                    -(a_safe * a_safe.log()).sum(dim=-1)
                    / math.log(float(c))
                )
            else:
                entropy_selected = torch.zeros(
                    selected_f.shape[:2],
                    device=selected_f.device,
                    dtype=selected_f.dtype,
                )

            if c >= 2:
                top2_prob, top2_idx = torch.topk(
                    assignment,
                    k=2,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
                p1 = top2_prob[..., 0]
                p2 = top2_prob[..., 1]
                idx1 = top2_idx[..., 0]
                idx2 = top2_idx[..., 1]

                # Relative Top-1/Top-2 probability margin.
                semantic_margin = (
                    (p1 - p2) / (p1 + p2 + self.eps)
                ).clamp(0.0, 1.0)

                # D4 soft-floor reliable ambiguity.
                #
                # D3's bell term is kept because it favors intermediate
                # ambiguity where the Top-1 direction is still informative.
                bell_reliability = (
                    4.0 * semantic_margin * (1.0 - semantic_margin)
                )

                # Weak recovery for extremely tied nodes.
                # It vanishes quadratically as semantics become clear, so
                # confident nodes still receive essentially no correction.
                low_margin_recovery = (
                    0.25 * (1.0 - semantic_margin).square()
                )

                reliable_ambiguity = (
                    bell_reliability + low_margin_recovery
                ).clamp(0.0, 1.0)

                proto_top1 = proto[idx1]  # [B,K,D]
                proto_top2 = proto[idx2]  # [B,K,D]

                # Discriminative semantic direction:
                # attraction to Top-1 + repulsion from Top-2.
                semantic_direction = F.normalize(
                    proto_top1 - proto_top2,
                    dim=-1,
                    eps=1e-6,
                ) * math.sqrt(float(d))
            else:
                semantic_margin = torch.ones(
                    selected_f.shape[:2],
                    device=selected_f.device,
                    dtype=selected_f.dtype,
                )
                bell_reliability = torch.zeros_like(semantic_margin)
                low_margin_recovery = torch.zeros_like(semantic_margin)
                reliable_ambiguity = torch.zeros_like(semantic_margin)
                semantic_direction = torch.zeros_like(selected_f)

            r_selected = torch.gather(
                refinement_score.float(),
                1,
                selected_indices,
            ).clamp(0.0, 1.0)

            gate_input = torch.cat(
                [
                    selected_f,
                    semantic_direction,
                    reliable_ambiguity.unsqueeze(-1),
                    r_selected.unsqueeze(-1),
                ],
                dim=-1,
            )

            # R is contextual evidence only.
            base_gate = torch.sigmoid(
                self.gate(gate_input)
            ).squeeze(-1)

            # D1 principle retained: exact zero-start, signed and bounded.
            semantic_residual_strength = torch.tanh(
                self.semantic_residual_raw.float()
            )

            effective_gate = (
                semantic_residual_strength
                * base_gate
                * reliable_ambiguity
            )

            # Identity preserving at alpha_sem=0.
            refined = (
                selected_f
                + effective_gate.unsqueeze(-1) * semantic_direction
            )

        raw_ambiguity = (1.0 - semantic_margin).clamp(0.0, 1.0)

        return refined, {
            "ls_scna_gate": effective_gate.detach(),
            "ls_scna_base_gate": base_gate.detach(),
            "ls_scna_assignment_entropy": entropy_selected.detach(),
            "ls_scna_residual_strength": semantic_residual_strength.detach(),
            "ls_scna_ambiguity": raw_ambiguity.detach(),
            "ls_scna_reliable_ambiguity": reliable_ambiguity.detach(),
            "ls_scna_bell_reliability": bell_reliability.detach(),
            "ls_scna_low_margin_recovery": low_margin_recovery.detach(),
            "ls_scna_top12_margin": semantic_margin.detach(),
        }



class SparseLocalRefinement(nn.Module):
    """
    Third-version global-guided sparse local refinement.

    Pipeline for CMRDE-selected nodes:
      fixed local spatial adjacency
        -> global-aware semantic edge gate
        -> Max-Relative GraphConv
        -> GELU
        -> optional LS-SCNA
        -> sparse scatter back
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_ls_scna: bool = True,
        ls_scna_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.dim = dim
        self.kernel_size = kernel_size
        hidden_dim = hidden_dim or dim

        # Edge gate is globally guided by original node, local difference,
        # Performer global increment and CMRDE refinement score R.
        gate_in = dim * 3 + 1
        self.edge_gate = nn.Sequential(
            nn.Linear(gate_in, max(dim // 2, 64)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 64), 1),
        )
        self.graph_conv = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.post_graph_act = nn.GELU()
        self.out_norm = nn.LayerNorm(dim)

        # D2 clean-ablation initialization:
        # constructing the extra D-only module must not shift the RNG state used
        # to initialize the downstream modules shared with C3.
        if use_ls_scna:
            with torch.random.fork_rng(devices=[]):
                self.ls_scna = LabelSemanticGuidedSCNA(
                    dim=dim,
                    temperature=ls_scna_temperature,
                    dropout=dropout,
                )
        else:
            self.ls_scna = None

    @staticmethod
    def _batched_gather_nodes(x: Tensor, indices: Tensor) -> Tensor:
        """Gather x[B,N,D] using indices[B,...] -> [B,...,D]."""
        b, n, d = x.shape
        flat_x = x.reshape(b * n, d)
        batch_offsets = torch.arange(
            b, device=x.device
        ).view(b, *([1] * (indices.dim() - 1))) * n
        flat_idx = (indices + batch_offsets).reshape(-1)
        gathered = flat_x[flat_idx]
        return gathered.reshape(*indices.shape, d)

    def forward(
        self,
        x: Tensor,
        global_nodes: Tensor,
        delta_global: Tensor,
        r: Tensor,
        height: int,
        width: int,
        selected_indices: Tensor,
        label_prototypes: Tensor,
        selected_active_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, n, d = x.shape
        if n != height * width:
            raise ValueError(f"N={n} does not equal H*W={height*width}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f=x.float(); global_f=global_nodes.float(); dg_f=delta_global.float(); r_f=r.float()
            neighbor_table=_build_local_neighbor_index(height,width,self.kernel_size,x.device)
            selected_neighbor_idx=neighbor_table[selected_indices]
            centers=self._batched_gather_nodes(x_f,selected_indices)
            neighbors=self._batched_gather_nodes(x_f,selected_neighbor_idx)
            global_delta=self._batched_gather_nodes(dg_f,selected_indices)
            r_selected=self._batched_gather_nodes(r_f.unsqueeze(-1),selected_indices)
            kn=neighbors.size(2)
            center_exp=centers.unsqueeze(2).expand(-1,-1,kn,-1); global_exp=global_delta.unsqueeze(2).expand(-1,-1,kn,-1); r_exp=r_selected.unsqueeze(2).expand(-1,-1,kn,-1)
            diff=neighbors-center_exp
            edge_gate=torch.sigmoid(self.edge_gate(torch.cat([center_exp,diff,global_exp,r_exp],dim=-1)))
            max_relative=(diff*edge_gate).max(dim=2).values
            delta_selected=self.graph_conv(torch.cat([centers,max_relative],dim=-1))
            refined_selected=centers+self.post_graph_act(delta_selected)

            # Preserve the exact C3 local-refinement path first.
            refined_selected = self.out_norm(refined_selected)

            local_diag={"mean_edge_gate":edge_gate.mean(dim=(2,3))}
            if self.ls_scna is not None:
                refined_selected,scna_diag=self.ls_scna(
                    refined_selected,
                    global_f,
                    label_prototypes,
                    selected_indices,
                    r_f,
                )
                local_diag.update(scna_diag)

            delta_selected=refined_selected-centers

            # C2 uses a rectangular Top-K_budget tensor for batched efficiency.
            # Entries beyond each image's adaptive K_b are padding candidates:
            # they must not contribute any local correction.
            if selected_active_mask is not None:
                active = selected_active_mask.to(
                    device=delta_selected.device,
                    dtype=delta_selected.dtype,
                )
                delta_selected = delta_selected * active.unsqueeze(-1)
                local_diag["adaptive_active_ratio"] = active.mean(dim=1)

            sparse_delta=torch.zeros((b,n,d),device=x.device,dtype=torch.float32)
            sparse_delta.scatter_(1,selected_indices.unsqueeze(-1).expand(-1,-1,d),delta_selected)
        return sparse_delta, local_diag



class SharedFFN(nn.Module):
    """Lightweight shared FFN D -> 2D -> D with residual connection."""

    def __init__(self, dim: int, expansion: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffn(self.norm(x))



# -----------------------------------------------------------------------------
# B1+B6: Label-Specific Global Readout + Prediction-Aware Aggregation
# -----------------------------------------------------------------------------


class LabelSpecificGlobalReadout(nn.Module):
    """Class-aware readout over globally enhanced graph nodes.

    Motivation
    ----------
    The B0 model compresses all N global nodes with ``mean(dim=1)`` before
    multi-label classification. That forces every label to share exactly the
    same pooled representation. B1 keeps the FAVOR+ global relation module
    unchanged, but replaces that single shared readout with C learnable label
    queries. Each label query attends to the N global nodes independently:

        A = softmax(Q K^T / sqrt(D))          [B, C, N]
        F_label = A V                         [B, C, D]

    A class-specific linear scorer converts each label feature into its own
    auxiliary GNN logit. For the downstream graph stream (C/D/E and the simple
    fusion used before DRCF), the label features are averaged and injected as a
    residual correction to the original global mean. The residual projection
    is zero-initialized, so B1 starts from the B0 global summary and learns how
    much class-aware readout information should be added.

    This module changes only the GLOBAL READOUT. It does not perform CNN-GNN
    fusion and therefore does not overlap with the later DRCF module.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.scale = self.dim ** -0.5

        self.label_queries = nn.Parameter(torch.empty(self.num_classes, self.dim))
        nn.init.trunc_normal_(self.label_queries, std=0.02)

        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.value_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.label_feature_norm = nn.LayerNorm(self.dim)

        # One classifier vector per label. This avoids mapping every label
        # feature to all C classes and then taking a diagonal.
        self.classifier_weight = nn.Parameter(torch.empty(self.num_classes, self.dim))
        self.classifier_bias = nn.Parameter(torch.zeros(self.num_classes))
        nn.init.xavier_uniform_(self.classifier_weight)

        # Preserve B0 at initialization: summary == mean(global_nodes).
        self.summary_proj = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.zeros_(self.summary_proj.weight)

        # B6: prediction-aware label aggregation. A single residual strength
        # controls how far the class-aware summary moves from the uniform
        # label average toward a confidence-weighted label average.
        # raw=0 -> strength=0 -> EXACTLY the B5/B1 aggregation at init.
        self.prediction_aggregation_raw = nn.Parameter(torch.zeros(1))

    def forward(self, nodes: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Args:
            nodes: globally enhanced nodes [B, N, D].

        Returns:
            summary: class-aware global summary [B, D] for downstream modules.
            logits: label-specific auxiliary GNN logits [B, C].
            diagnostics: lightweight attention statistics.
        """
        if nodes.dim() != 3 or nodes.size(-1) != self.dim:
            raise ValueError(
                f"LabelSpecificGlobalReadout expects [B,N,{self.dim}], "
                f"got {tuple(nodes.shape)}"
            )

        # Keep the small attention/readout block in FP32 for numerical stability.
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            x = nodes.float()
            q = self.query_norm(self.label_queries.float())               # [C,D]
            k = self.key_norm(x)                                          # [B,N,D]
            v = self.value_proj(x)                                        # [B,N,D]

            scores = torch.einsum("cd,bnd->bcn", q, k) * self.scale      # [B,C,N]
            attention = torch.softmax(scores, dim=-1)
            attention = self.attn_dropout(attention)

            label_features = torch.einsum("bcn,bnd->bcd", attention, v)  # [B,C,D]
            label_features = self.label_feature_norm(
                label_features + q.unsqueeze(0)
            )

            logits = (
                torch.einsum("bcd,cd->bc", label_features, self.classifier_weight.float())
                * self.scale
                + self.classifier_bias.float()
            )

            # The B0 readout is kept as a stable residual anchor. B6 keeps the
            # original uniform label average, then learns a residual move toward
            # a prediction-aware label aggregation. This changes only the GLOBAL
            # branch internal summary; no CNN feature/logit participates here.
            mean_nodes = x.mean(dim=1)
            mean_label_feature = label_features.mean(dim=1)

            # Per-label presence confidence from the supervised GNN logits.
            # Detach the weights for aggregation stability: the GNN logits remain
            # directly supervised by BCE, while final-loss gradients do not form
            # a self-reinforcing feedback loop through the confidence weights.
            label_confidence = torch.sigmoid(logits).detach()             # [B,C]
            confidence_sum = label_confidence.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weighted_label_feature = torch.sum(
                label_features * label_confidence.unsqueeze(-1), dim=1
            ) / confidence_sum

            # Smooth bounded residual strength with exact identity at raw=0.
            # Positive values move toward prediction-aware aggregation; negative
            # values are allowed during exploration and indicate the hypothesis is
            # not supported by the data. Range is approximately [-0.5, 0.5].
            aggregation_strength = 0.5 * torch.tanh(
                self.prediction_aggregation_raw.float()
            )
            aggregated_label_feature = (
                mean_label_feature
                + aggregation_strength
                * (weighted_label_feature - mean_label_feature)
            )
            summary = mean_nodes + self.summary_proj(aggregated_label_feature)

            # Normalized entropy: 1 = diffuse/uniform, 0 = concentrated.
            if x.size(1) > 1:
                a = attention.clamp_min(1e-8)
                entropy = -(a * a.log()).sum(dim=-1) / math.log(float(x.size(1)))
            else:
                entropy = torch.zeros(
                    (x.size(0), self.num_classes), device=x.device, dtype=x.dtype
                )
            peak = attention.amax(dim=-1)

        # Confidence-distribution diagnostics for B6. Normalize across labels
        # only for entropy reporting; the actual aggregation uses raw sigmoid
        # confidences normalized by their sum as defined above.
        conf_dist = label_confidence / label_confidence.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        if self.num_classes > 1:
            cdist = conf_dist.clamp_min(1e-8)
            confidence_entropy = -(cdist * cdist.log()).sum(dim=1) / math.log(
                float(self.num_classes)
            )
        else:
            confidence_entropy = torch.zeros(
                (nodes.size(0),), device=nodes.device, dtype=torch.float32
            )

        diagnostics = {
            "label_readout_entropy": entropy.detach(),
            "label_readout_peak": peak.detach(),
            "prediction_aggregation_strength": aggregation_strength.detach(),
            "label_confidence_mean": label_confidence.mean(dim=1).detach(),
            "label_confidence_entropy": confidence_entropy.detach(),
        }
        return summary, logits, diagnostics

# -----------------------------------------------------------------------------
# Legacy E0 DRCF + E2 Unidirectional Graph Relation Correction
# -----------------------------------------------------------------------------


class DRCF(nn.Module):
    """
    Legacy E0 Dual-Reliability Cross-Enhancement Fusion.

    This class is retained ONLY to reproduce the original E0 RNG consumption
    before the B/C/D ``simple_fusion`` module is initialized. It is never
    registered as a submodule and never used in the E2 forward pass.
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        rel_hidden = max(dim // 4, 64)
        self.cnn_reliability = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, rel_hidden),
            nn.GELU(),
            nn.Linear(rel_hidden, 1),
        )
        self.gnn_reliability = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, rel_hidden),
            nn.GELU(),
            nn.Linear(rel_hidden, 1),
        )

        gate_in = dim * 4 + 2
        gate_hidden = max(dim, 128)
        self.g_to_c_gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, dim),
        )
        self.c_to_g_gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, dim),
        )
        self.g_to_c_proj = nn.Linear(dim, dim, bias=False)
        self.c_to_g_proj = nn.Linear(dim, dim, bias=False)
        self.c_norm = nn.LayerNorm(dim)
        self.g_norm = nn.LayerNorm(dim)
        self.complementary_fusion = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, c: Tensor, g: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise RuntimeError("Legacy DRCF is RNG-compatibility only in E2.")


class GraphRelationCorrection(nn.Module):
    """
    E3 backbone: Unidirectional Graph Relation Correction (UGRC).

    Motivation from E0/E1 diagnostics
    ---------------------------------
    The original DRCF learned nearly saturated image-level reliability:
    q_C -> 0, q_G -> 1, alpha -> 1. In the 0.25-gradient diagnostic the GNN
    branch became very strong while the final DRCF prediction was worse than
    that branch, indicating over-correction rather than insufficient graph
    evidence.

    E2 therefore removes:
      1) q_C / q_G reliability estimators,
      2) bidirectional G->C and C->G feature enhancement,
      3) complementary four-way fusion.

    Instead, CNN remains the visual decision anchor and GNN supplies only a
    relation-correction representation. The CNN relation feature is treated as
    a stop-gradient reference inside this correction path, so the correction
    branch cannot introduce an extra backward path into the CNN representation.
    The GNN branch remains trainable through the correction path.

        c_ref = stopgrad(LN(C))
        g     = LN(G)
        gap   = g - c_ref
        h_rel = MLP([g, gap])
        h_rel = MLP([g, gap])
        Delta_z_graph = Classifier(h_rel)

    The coarse E2 global beta is removed in E3. Its role is replaced by the
    proposed label-wise correction-demand matrix R_F in [B,C].
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        gnn_correction_grad_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.gnn_correction_grad_scale = float(gnn_correction_grad_scale)
        if not 0.0 <= self.gnn_correction_grad_scale <= 1.0:
            raise ValueError(
                "gnn_correction_grad_scale must be in [0, 1], "
                f"got {self.gnn_correction_grad_scale}"
            )

        self.c_ref_norm = nn.LayerNorm(dim)
        self.g_norm = nn.LayerNorm(dim)

        # Lightweight one-way relation correction: [G, G-C_ref] -> D.
        self.relation_correction = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # No image-level reliability scalar is used in E3.
        # The external correction classifier remains zero-start, so the
        # complete E3 model still begins exactly from CNN logits.

    def forward(self, c: Tensor, g: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if c.shape != g.shape:
            raise ValueError(
                f"CNN/GNN relation features must match, got {tuple(c.shape)} "
                f"and {tuple(g.shape)}"
            )

        with torch.autocast(device_type=c.device.type, enabled=False):
            c_f = c.float()
            g_f = g.float()

            # Non-invasive visual reference:
            # CNN values still participate in the correction forward pass, but
            # this branch sends no correction-path gradient into the CNN branch.
            # Detach BEFORE LayerNorm so the correction module's LayerNorm
            # parameters remain trainable.
            c_ref = self.c_ref_norm(c_f.detach())

            # E2.1: weakly-coupled GNN correction gradient.
            #
            # Forward value is exactly g_f:
            #   g_corr = sg(g_f) + s * (g_f - sg(g_f))
            # but backward gradient into the upstream GNN is scaled by s.
            #
            # s=0.25 keeps a small amount of useful joint supervision while
            # preventing the correction objective from dominating the GNN branch.
            g_detached = g_f.detach()
            g_corr = (
                g_detached
                + self.gnn_correction_grad_scale * (g_f - g_detached)
            )
            g_rel = self.g_norm(g_corr)

            signed_gap = g_rel - c_ref
            correction_feature = self.relation_correction(
                torch.cat([g_rel, signed_gap], dim=-1)
            )

            # Diagnostics only; label-wise gating is handled by the E3
            # correction-demand estimator after CNN/GNN logits are available.
            gap_norm = torch.linalg.vector_norm(
                signed_gap, dim=-1, keepdim=True
            ) / math.sqrt(float(self.dim))

        return correction_feature, {
            "relation_gap_norm": gap_norm.detach(),
            "gnn_correction_grad_scale": torch.tensor(
                self.gnn_correction_grad_scale,
                device=correction_feature.device,
                dtype=correction_feature.dtype,
            ),
        }


class LabelWiseCorrectionDemand(nn.Module):
    """
    E4: Label-wise Adaptive Branch Anchor + Correction Demand.

    E3 showed that the GNN branch can become clearly stronger than the CNN branch
    (e.g. 84.50 vs 83.33 mAP), while a fixed CNN anchor can still limit the final
    prediction. E4 therefore keeps E3's label-wise correction-demand principle,
    but removes the assumption that CNN must always be the decision anchor.

    For each label i:
        pC_i = sigmoid(zC_i)
        pG_i = sigmoid(zG_i)
        confC_i = |2 pC_i - 1|
        confG_i = |2 pG_i - 1|

    A parameter-free relative-confidence soft anchor is computed:
        lambdaG_i =
            exp(confG_i / tau)
            ---------------------------------
            exp(confC_i / tau) + exp(confG_i / tau)

    tau is one shared global constant (default 0.25), not class-specific and not
    dataset-specific.

    The adaptive base prediction is:
        zBase_i = (1-lambdaG_i) * zCNN_i + lambdaG_i * zGNN_i

    E3 correction demand is retained:
        U_C_i = 1-confC_i
        U_G_i = 1-confG_i
        D_i   = |pC_i-pG_i|
        S_i   = 0.5 * (1 + confG_i-confC_i)
        A_i   = sigmoid(SharedMLP([U_C,U_G,D,S,D*S]))
        R_i^F = D_i * S_i * A_i

    Final E4.1 prediction:
        zFinal_i = zBase_i + R_i^F * Delta z_i^relation

    E4.1 keeps the E4 forward computation exactly unchanged, but protects the
    upstream GNN from the adaptive-base loss path:
        forward(zGNN_base) = zGNN
        d zGNN_base / d zGNN = 0

    The relation-correction path still keeps the proven weak GNN gradient
    controlled by gnn_correction_grad_scale (default 0.25).

    Thus:
      - a stronger/more confident CNN automatically receives more base weight;
      - a stronger/more confident GNN automatically receives more base weight;
      - only labels with actual disagreement and graph support receive extra
        relation correction;
      - all labels share the same demand estimator.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        dropout: float = 0.0,
        anchor_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(hidden_dim), 8)
        self.anchor_temperature = float(anchor_temperature)
        if self.anchor_temperature <= 0:
            raise ValueError("anchor_temperature must be positive")

        self.shared_gate = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cnn_logits: Tensor,
        gnn_logits: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if cnn_logits.shape != gnn_logits.shape:
            raise ValueError(
                "cnn_logits and gnn_logits must have identical [B,C] shapes, "
                f"got {tuple(cnn_logits.shape)} and {tuple(gnn_logits.shape)}"
            )

        with torch.autocast(device_type=cnn_logits.device.type, enabled=False):
            # Demand evidence is diagnostic/control information only. Detaching
            # prevents another final-loss gradient route into CNN/GNN branches.
            z_c = cnn_logits.detach().float()
            z_g = gnn_logits.detach().float()

            p_c = torch.sigmoid(z_c)
            p_g = torch.sigmoid(z_g)

            conf_c = torch.abs(2.0 * p_c - 1.0).clamp(0.0, 1.0)
            conf_g = torch.abs(2.0 * p_g - 1.0).clamp(0.0, 1.0)
            u_c = (1.0 - conf_c).clamp(0.0, 1.0)
            u_g = (1.0 - conf_g).clamp(0.0, 1.0)

            disagreement = torch.abs(p_c - p_g).clamp(0.0, 1.0)

            # E4 adaptive label-wise branch anchor.
            # Softmax over relative confidence gives a bounded preference:
            #   0.5 -> comparable confidence
            #   >0.5 -> GNN is more confident
            #   <0.5 -> CNN is more confident
            branch_confidence = torch.stack([conf_c, conf_g], dim=-1)
            branch_weight = torch.softmax(
                branch_confidence / self.anchor_temperature,
                dim=-1,
            )
            graph_anchor_weight = branch_weight[..., 1]

            # >0.5 when GNN is more confident, <0.5 when CNN is more confident.
            graph_support = (
                0.5 * (1.0 + conf_g - conf_c)
            ).clamp(0.0, 1.0)

            base_demand = disagreement * graph_support

            evidence = torch.stack(
                [
                    u_c,
                    u_g,
                    disagreement,
                    graph_support,
                    base_demand,
                ],
                dim=-1,
            )  # [B,C,5]

            learned_gate = torch.sigmoid(
                self.shared_gate(evidence).squeeze(-1)
            )

            # Structural demand gate. Even if learned_gate -> 1 globally,
            # agreement labels still receive ~0 correction.
            correction_demand = (
                base_demand * learned_gate
            ).clamp(0.0, 1.0)

        diagnostics = {
            "label_correction_demand": correction_demand.detach(),
            "label_disagreement": disagreement.detach(),
            "label_graph_support": graph_support.detach(),
            "label_graph_anchor_weight": graph_anchor_weight.detach(),
            "label_cnn_uncertainty": u_c.detach(),
            "label_gnn_uncertainty": u_g.detach(),
            "label_demand_learned_gate": learned_gate.detach(),
        }
        return correction_demand, diagnostics



# -----------------------------------------------------------------------------
# Chapter-5 CP-ASRC: complementarity-preserving adaptive fusion
# -----------------------------------------------------------------------------


class ComplementarityPreservingFusion(nn.Module):
    """Consensus-disagreement fusion that generalizes the successful 0.5/0.5 rule.

    Fixed fusion works well because it (1) gives both branches an equal direct
    route to the decision and (2) lets the main loss jointly shape CNN/GNN into
    complementary predictors.  CP-ASRC keeps that consensus as the center and
    only learns a small label-wise displacement along the *existing* CNN--GNN
    disagreement direction:

        z_cons = 0.5 * (z_C + z_G)
        d      = 0.5 * (z_G - z_C)
        beta   = beta_max * tanh(router(evidence))
        z      = z_cons + beta * d

    Therefore the effective weights are
        w_C = 0.5 * (1 - beta),  w_G = 0.5 * (1 + beta).

    The final layers of both router paths are zero-initialized, so beta == 0
    exactly at initialization and CP-ASRC starts from the fixed-fusion model.
    Router evidence is detached from the upstream branches; the *fusion logits*
    themselves are not detached, so the main loss still jointly optimizes CNN
    and GNN through one unified fusion equation.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        beta_max: float = 0.20,
    ) -> None:
        super().__init__()
        d = int(feature_dim)
        c = int(num_classes)
        h = int(hidden_dim)
        self.beta_max = float(beta_max)
        if not (0.0 < self.beta_max < 1.0):
            raise ValueError("beta_max must be in (0,1)")

        self.c_norm = nn.LayerNorm(d)
        self.g_norm = nn.LayerNorm(d)
        self.feature_router = nn.Sequential(
            nn.Linear(d * 4, h, bias=False),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, c),
        )
        self.evidence_router = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Exact fixed-fusion start.  Earlier variants added a second residual
        # after the consensus; here the adaptive mechanism itself *is* fusion.
        nn.init.zeros_(self.feature_router[-1].weight)
        nn.init.zeros_(self.feature_router[-1].bias)
        nn.init.zeros_(self.evidence_router[-1].weight)
        nn.init.zeros_(self.evidence_router[-1].bias)

    def forward(
        self,
        cnn_feature: Tensor,
        gnn_feature: Tensor,
        cnn_logits: Tensor,
        gnn_logits: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if cnn_logits.shape != gnn_logits.shape:
            raise ValueError(
                f"CNN/GNN logits must match, got {tuple(cnn_logits.shape)} and {tuple(gnn_logits.shape)}"
            )

        # Routing evidence does not create a hidden gradient path into either
        # representation branch.  Branch interaction happens transparently in
        # the final weighted fusion below.
        c = self.c_norm(cnn_feature.detach().float())
        g = self.g_norm(gnn_feature.detach().float())
        feature_desc = torch.cat([c, g, g - c, c * g], dim=-1)
        feature_score = self.feature_router(feature_desc)

        zc = cnn_logits.detach().float()
        zg = gnn_logits.detach().float()
        pc = torch.sigmoid(zc)
        pg = torch.sigmoid(zg)
        u_c = 4.0 * pc * (1.0 - pc)
        u_g = 4.0 * pg * (1.0 - pg)
        prob_gap = pg - pc
        abs_gap = prob_gap.abs()
        conf_margin = torch.tanh(zg.abs() - zc.abs())
        evidence = torch.stack(
            [pc, pg, abs_gap, u_c, u_g, conf_margin], dim=-1
        )
        evidence_score = self.evidence_router(evidence).squeeze(-1)

        beta = self.beta_max * torch.tanh(feature_score + evidence_score)
        beta = beta.to(cnn_logits.dtype)

        consensus = 0.5 * (cnn_logits + gnn_logits)
        disagreement = 0.5 * (gnn_logits - cnn_logits)
        fused = consensus + beta * disagreement

        cnn_weight = 0.5 * (1.0 - beta)
        gnn_weight = 0.5 * (1.0 + beta)
        aux = {
            "cp_consensus_logits": consensus,
            "cp_beta": beta,
            "cp_beta_abs_mean": beta.detach().abs().mean(dim=-1),
            "cp_beta_min": beta.detach().amin(dim=-1),
            "cp_beta_max": beta.detach().amax(dim=-1),
            "cp_cnn_weight_mean": cnn_weight.detach().mean(dim=-1),
            "cp_gnn_weight_mean": gnn_weight.detach().mean(dim=-1),
            "cp_probability_disagreement": abs_gap.detach().mean(dim=-1),
        }
        return fused, aux


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------


@dataclass
class GLRDRCFConfig:
    num_classes: int
    embed_dim: int = 256
    num_heads: int = 4
    favor_features: int = 64
    favor_orthogonal_scaling: int = 0
    refine_ratio: float = 0.25
    min_refine_nodes: int = 4
    local_kernel_size: int = 3
    ls_scna_temperature: float = 0.2
    dropout: float = 0.1
    pretrained_backbone: bool = True
    backbone_weights: str = ""
    use_cmrde: bool = True
    use_local_refinement: bool = True
    use_drcf: bool = True
    use_ls_scna: bool = True
    gnn_correction_grad_scale: float = 0.25
    # Chapter-5 V14: only C enables the gradient-isolated Lite-SADRF sidecar.
    use_lite_sadrf: bool = False
    lite_sadrf_hidden: int = 48
    lite_sadrf_delta_max: float = 0.30
    # Fusion modes:
    #   full=Chapter-4 ASRC
    #   fixed=0.5/0.5 fixed fusion
    #   consensus_asrc=previous CA-ASRC diagnostic variant
    #   decoupled_asrc=previous DCA-ASRC diagnostic variant
    #   cp_asrc=CP-ASRC: one unified consensus-disagreement adaptive fusion
    fusion_mode: str = "full"
    dca_residual_max: float = 0.25
    dca_scale_init: float = 0.05
    cp_beta_max: float = 0.20
    cp_hidden_dim: int = 64


class GLRDRCFNet(nn.Module):
    """
    Semantic-aware residual GLR-DRCFNet.

    Main design changes relative to the original third version:
    -----------------------------------------------------------
    1) Fine graph nodes come from ResNet101 Layer3 instead of Layer4.
    2) Layer4 is preserved as the high-level CNN semantic anchor.
    3) B5 calibrates FAVOR+ residual updates with its own relation uncertainty
       before the proven B1 label-specific global readout.
    4) CMRDE uncertainty demand is filtered by node semantic relevance:
           R_route = R_CMRDE * A_sem
    5) C1 uses R_route for local edge context, but decouples final local
       residual magnitude:
           X_graph = X_global + alpha_local * Delta_local
           alpha_local = sigmoid(s_local)
    6) C2 replaces fixed-ratio Top-K with parameter-free image-adaptive routing:
           tau_b = mean(R_b) + std(R_b)
           K_b = clamp(count(R_b >= tau_b), min_nodes, K_budget)
       where K_budget is the original refine_ratio budget (default 25%).
       Thus difficult images may use more of the budget, while easy images
       avoid unnecessary local refinement.
    7) C3 makes local correction conservative at the start of training:
           alpha_local = tanh(s_local),  s_local initialized to 0
           X_graph = X_global + alpha_local * Delta_local
       Therefore the local branch contributes exactly zero at initialization
       and is introduced only when optimization finds a useful correction.
    8) E2 removes bidirectional DRCF and uses one-way graph relation correction:
           logits_final = logits_CNN + beta * delta_logits_graph
       where beta is one bounded global calibration scalar.
    9) The correction classifier is zero-initialized, so the complete model
       starts exactly from the CNN baseline prediction and learns corrections.
    """

    def __init__(self, config: GLRDRCFConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(config.num_classes)
        self.embed_dim = int(config.embed_dim)

        # ------------------------------------------------------------------
        # Standard CNN. Architecture remains unchanged.
        # ------------------------------------------------------------------
        self.backbone = ResNet101FeatureExtractor(
            pretrained=config.pretrained_backbone,
            local_weights=config.backbone_weights,
        )

        # Layer3 -> fine graph nodes. At 224x224 input this is normally 14x14.
        self.node_proj = nn.Conv2d(
            ResNet101FeatureExtractor.layer3_channels,
            self.embed_dim,
            kernel_size=1,
            bias=False,
        )
        self.node_proj_norm = nn.LayerNorm(self.embed_dim)

        # Global relation modeling is unchanged: standard FAVOR+ Performer.
        self.global_relation = FAVORPlusPerformerAttention(
            dim=self.embed_dim,
            num_heads=config.num_heads,
            num_features=config.favor_features,
            orthogonal_scaling=config.favor_orthogonal_scaling,
            proj_dropout=config.dropout,
        )

        # B5: use FAVOR+'s own relation uncertainty to calibrate only the
        # global relation residual. Zero initialization makes this an exact B1
        # identity at the start of training.
        self.relation_calibration = RelationConfidenceResidualCalibration()

        # Layer4 semantic context is projected only to GUIDE node relevance and
        # DRCF. It does not alter the internal ResNet101 architecture.
        self.semantic_context_proj = nn.Sequential(
            nn.Linear(ResNet101FeatureExtractor.out_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )

        # Node-wise semantic head provides U_cls, auxiliary supervision and
        # label-semantic relevance A_sem.
        self.node_semantic_head = nn.Linear(self.embed_dim, self.num_classes)
        self.cmrde = CMRDE(
            hidden_dim=max(self.embed_dim // 4, 64),
            dropout=config.dropout,
        )

        self.local_refinement = SparseLocalRefinement(
            dim=self.embed_dim,
            kernel_size=config.local_kernel_size,
            dropout=config.dropout,
            use_ls_scna=config.use_ls_scna,
            ls_scna_temperature=config.ls_scna_temperature,
        )

        # C1/C3: Demand-Strength Decoupling + Zero-Start Residual.
        #
        # R remains responsible for routing and local edge context, while the
        # final local correction has an independent shared residual strength.
        #
        # C3 uses a ReZero-style signed scalar:
        #
        #     alpha_local = tanh(local_residual_raw) in (-1, 1)
        #     X_out = X_global + alpha_local * Delta_local
        #
        # local_residual_raw=0 -> alpha_local=0 EXACTLY.
        # Thus C3 begins from the B6 global representation and introduces local
        # correction only after the optimizer finds a beneficial direction.
        #
        # torch.zeros is deterministic and consumes no RNG, so B6's existing
        # random initialization sequence is preserved under the same seed.
        self.local_residual_raw = nn.Parameter(torch.zeros(()))

        self.graph_fusion_norm = nn.LayerNorm(self.embed_dim)
        self.shared_ffn = SharedFFN(
            self.embed_dim,
            expansion=2.0,
            dropout=config.dropout,
        )

        # B1 readout retained in B5: label-specific global semantic readout.
        self.global_readout = LabelSpecificGlobalReadout(
            num_classes=self.num_classes,
            dim=self.embed_dim,
            dropout=config.dropout,
        )

        # ------------------------------------------------------------------
        # CNN semantic anchor.
        # ------------------------------------------------------------------
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # This is the preserved pure-CNN prediction path:
        # Layer4 (2048-D) -> Linear(num_classes).
        self.cnn_baseline_classifier = nn.Linear(
            ResNet101FeatureExtractor.out_channels,
            self.num_classes,
        )

        # A separate projection is used only for CNN-GNN interaction in DRCF.
        self.cnn_relation_proj = nn.Sequential(
            nn.Linear(ResNet101FeatureExtractor.out_channels, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        self.gnn_pool_norm = nn.LayerNorm(self.embed_dim)

        # ------------------------------------------------------------------
        # Fusion / correction.
        #
        # Preserve the exact RNG state seen by the existing B/C/D simple-fusion
        # module. The original E0 DRCF used to be initialized before it, so we
        # reproduce that RNG consumption with a temporary unregistered object.
        # This keeps B/C/D initialization comparable if they are rerun.
        # ------------------------------------------------------------------
        _legacy_rng_compat = DRCF(self.embed_dim, dropout=config.dropout)

        # B/C/D baseline integration remains unchanged.
        self.simple_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        del _legacy_rng_compat

        # E2 / Chapter-5 relation-correction module.  For DCA-ASRC the
        # correction branch must not perturb the stochastic or gradient trajectory
        # of the strong fixed-fusion CNN/GNN branches. Therefore DCA uses:
        #   * dropout=0 inside the correction path (no extra RNG consumption),
        #   * upstream GNN gradient scale=0 (features are correction evidence only).
        # Module initialization is forked so shared CNN/GNN initialization remains identical.
        _dca_mode = str(config.fusion_mode).lower() == "decoupled_asrc"
        _cp_mode = str(config.fusion_mode).lower() == "cp_asrc"
        with torch.random.fork_rng(devices=[]):
            self.relation_correction = GraphRelationCorrection(
                self.embed_dim,
                dropout=(0.0 if _dca_mode else config.dropout),
                gnn_correction_grad_scale=(0.0 if _dca_mode else config.gnn_correction_grad_scale),
            )
            self.label_correction_demand = LabelWiseCorrectionDemand(
                hidden_dim=max(self.num_classes, 16),
                dropout=0.0,
            )

        # Relation residual head starts at zero so DCA-ASRC initially equals the
        # fixed consensus exactly.
        self.correction_classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.zeros_(self.correction_classifier.weight)
        nn.init.zeros_(self.correction_classifier.bias)

        # DCA-ASRC learns a conservative label-wise correction scale.  It starts
        # at dca_scale_init and is bounded by dca_residual_max; no random init is
        # used, so this also preserves the base-branch RNG trajectory.
        self.dca_scale_raw: Optional[nn.Parameter] = None
        if _dca_mode:
            max_scale = float(config.dca_residual_max)
            init_scale = float(config.dca_scale_init)
            if not (0.0 < init_scale < max_scale):
                raise ValueError("dca_scale_init must be in (0, dca_residual_max)")
            ratio = init_scale / max_scale
            init_raw = math.log(ratio / (1.0 - ratio))
            self.dca_scale_raw = nn.Parameter(
                torch.full((self.num_classes,), float(init_raw))
            )

        self.cp_fusion: Optional[ComplementarityPreservingFusion] = None
        if _cp_mode:
            with torch.random.fork_rng(devices=[]):
                self.cp_fusion = ComplementarityPreservingFusion(
                    feature_dim=self.embed_dim,
                    num_classes=self.num_classes,
                    hidden_dim=int(config.cp_hidden_dim),
                    beta_max=float(config.cp_beta_max),
                )

        # In M3/M4 the fusion is deliberately parameter-free.  Remove the
        # unused ASRC/simple-fusion modules after all shared CNN/GNN modules have
        # been initialized so Params reflects the actual controlled model while
        # preserving identical shared initialization under the same seed.
        if str(config.fusion_mode).lower() in {"fixed", "cp_asrc"}:
            self.simple_fusion = nn.Identity()
            self.relation_correction = nn.Identity()
            self.label_correction_demand = nn.Identity()
            self.correction_classifier = nn.Identity()

        # Chapter-5 V14 sidecar is instantiated only AFTER every shared SGRC
        # module. fork_rng guarantees B/C shared initialization remains identical.
        self.lite_sadrf: Optional[LiteSADRFSemanticResidualAdapter] = None
        if bool(config.use_lite_sadrf):
            with torch.random.fork_rng(devices=[]):
                self.lite_sadrf = LiteSADRFSemanticResidualAdapter(
                    num_classes=self.num_classes,
                    hidden_channels=int(config.lite_sadrf_hidden),
                    delta_max=float(config.lite_sadrf_delta_max),
                )

        # B1 global auxiliary logits are produced directly by the label-specific
        # readout, so a separate shared-vector GNN classifier is no longer needed.

    # ------------------------------------------------------------------
    # Uncertainty / semantic routing
    # ------------------------------------------------------------------

    def _build_uncertainties(
        self,
        global_nodes: Tensor,
        relation_uncertainty: Tensor,
        semantic_context: Tensor,
        height: int,
        width: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Build the third-version uncertainty evidence plus semantic relevance.

        U_rel/U_cls/U_local remain the CMRDE inputs.
        Layer4 semantic context guides the node-semantic response used to
        compute A_sem without changing the CNN itself.
        """
        with torch.autocast(device_type=global_nodes.device.type, enabled=False):
            g = global_nodes.float()
            context = semantic_context.float().unsqueeze(1)

            # Layer4-guided node semantics. The residual addition keeps local
            # node identity while injecting image-level high-level semantics.
            semantic_nodes = g + context
            node_logits = self.node_semantic_head(semantic_nodes)

            # Stable Bernoulli entropy from logits:
            # H(sigmoid(z)) = softplus(z) - z*sigmoid(z)
            p = torch.sigmoid(node_logits)
            entropy = (F.softplus(node_logits) - node_logits * p) / math.log(2.0)
            u_cls = entropy.mean(dim=-1).clamp(0.0, 1.0).detach()

            b, n, d = g.shape
            fmap = g.transpose(1, 2).reshape(b, d, height, width)
            local_mean = F.avg_pool2d(
                fmap,
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=False,
            ).flatten(2).transpose(1, 2)
            node_unit = F.normalize(g, dim=-1, eps=1e-6)
            local_unit = F.normalize(local_mean, dim=-1, eps=1e-6)
            u_local = (
                0.5 * torch.linalg.vector_norm(node_unit - local_unit, dim=-1)
            ).clamp(0.0, 1.0).detach()

            u_rel = relation_uncertainty.float().clamp(0.0, 1.0).detach()

            # Semantic relevance: a node is worth refining only when at least
            # one supervised label considers it relevant. This is a routing
            # statistic, so detach it just like the uncertainty evidence.
            semantic_relevance = p.amax(dim=-1).clamp(0.0, 1.0).detach()

        return u_rel, u_cls, u_local, node_logits, semantic_relevance

    def _select_refinement_nodes(
        self,
        r: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """C2 image-adaptive budgeted sparse routing.

        The original ``refine_ratio`` is preserved as a hard compute budget
        rather than a fixed per-image selection ratio.

        For each image b:
            tau_b = mean(R_b) + std(R_b)
            K_b   = count(R_b >= tau_b)
            K_b   = clamp(K_b, min_refine_nodes, K_budget)

        A rectangular Top-K_budget tensor is returned for efficient batched
        local refinement together with an ``active_mask`` that activates only
        the first K_b ranked nodes of each image.

        No learnable routing parameter or dataset-specific threshold is added.
        """
        if r.dim() != 2:
            raise ValueError(
                f"refinement score must be [B,N], got {tuple(r.shape)}"
            )

        b, n = r.shape
        budget_k = max(
            self.config.min_refine_nodes,
            int(round(n * self.config.refine_ratio)),
        )
        budget_k = min(max(budget_k, 1), n)

        with torch.no_grad():
            r_detached = r.detach().float()
            mean = r_detached.mean(dim=1)
            std = r_detached.std(dim=1, unbiased=False)
            threshold = mean + std

            requested_k = (r_detached >= threshold.unsqueeze(1)).sum(dim=1)
            requested_k = requested_k.clamp(
                min=min(self.config.min_refine_nodes, n),
                max=budget_k,
            )

        # Sort descending so rank < K_b directly defines the per-image active set.
        selected = torch.topk(
            r,
            k=budget_k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        ranks = torch.arange(
            budget_k,
            device=r.device,
        ).unsqueeze(0).expand(b, -1)
        active_mask = ranks < requested_k.unsqueeze(1)
        dynamic_ratio = requested_k.float() / float(n)

        return selected, active_mask, dynamic_ratio, threshold

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        # Standard ResNet101 features. Layer3 is graph source; Layer4 is the
        # high-level CNN semantic anchor and baseline classification source.
        layer2_map, layer3_map, layer4_map = self.backbone.forward_hierarchy(images)
        b = images.size(0)

        # ---------------- Fine-grained graph branch (Layer3) ----------------
        node_map = self.node_proj(layer3_map)
        _, _, h, w = node_map.shape
        nodes = node_map.flatten(2).transpose(1, 2)  # [B, H*W, D]
        nodes = self.node_proj_norm(nodes)
        pos = build_2d_sincos_position_encoding(
            h,
            w,
            self.embed_dim,
            nodes.device,
            nodes.dtype,
        )
        nodes_with_pos = nodes + pos

        raw_global_nodes, u_rel_raw, attn_diag = self.global_relation(nodes_with_pos)

        # B5: confidence-calibrated FAVOR+ residual. At initialization the
        # learned strength is exactly zero, so global_nodes == raw_global_nodes
        # and this forward path is identical to B1.
        global_nodes, calibration_diag = self.relation_calibration(
            base_nodes=nodes_with_pos,
            raw_global_nodes=raw_global_nodes,
            relation_uncertainty=u_rel_raw,
        )
        delta_global = global_nodes - nodes_with_pos

        # ---------------- CNN semantic anchor (Layer4) ----------------
        # V14 keeps the complete SGRC path pure.  The sidecar reads detached
        # Layer3/Layer4 descriptors and predicts a visual residual, but neither
        # SGLR-GNN nor CMRDE/ASRC is rewritten by that residual.
        pure_cnn_raw_feature = self.cnn_pool(layer4_map).flatten(1)  # [B,2048]
        pure_cnn_logits = self.cnn_baseline_classifier(pure_cnn_raw_feature)
        enhanced_cnn_logits = pure_cnn_logits
        cnn_visual_delta = torch.zeros_like(pure_cnn_logits)
        final_visual_delta = torch.zeros_like(pure_cnn_logits)
        lite_aux: Dict[str, Tensor] = {}

        # Internal SGRC uses the untouched CNN prediction.
        cnn_logits = pure_cnn_logits
        cnn_raw_feature = pure_cnn_raw_feature
        semantic_context = self.semantic_context_proj(pure_cnn_raw_feature)

        # ---------------- Semantic-aware CMRDE routing ----------------
        u_rel, u_cls, u_local, node_logits, semantic_relevance = (
            self._build_uncertainties(
                global_nodes=global_nodes,
                relation_uncertainty=u_rel_raw,
                semantic_context=semantic_context,
                height=h,
                width=w,
            )
        )

        if self.config.use_cmrde:
            r_cmrde, evidence_weights = self.cmrde(u_rel, u_cls, u_local)
            # Key modification:
            # uncertainty is necessary, semantic relevance decides whether the
            # uncertain region is actually worth refinement.
            r = (r_cmrde * semantic_relevance).clamp(0.0, 1.0)
        else:
            # Ablation: without CMRDE every node has equal refinement demand.
            r_cmrde = torch.ones_like(u_rel)
            r = torch.ones_like(u_rel)
            evidence_weights = torch.full(
                (*u_rel.shape, 3),
                1.0 / 3.0,
                device=u_rel.device,
                dtype=u_rel.dtype,
            )

        # ---------------- Sparse local refinement ----------------
        if self.config.use_local_refinement:
            if self.config.use_cmrde:
                (
                    selected,
                    selected_active_mask,
                    dynamic_refine_ratio,
                    adaptive_threshold,
                ) = self._select_refinement_nodes(r)
            else:
                selected = torch.arange(
                    nodes.size(1),
                    device=nodes.device,
                ).unsqueeze(0).expand(b, -1)
                selected_active_mask = torch.ones(
                    selected.shape,
                    device=nodes.device,
                    dtype=torch.bool,
                )
                dynamic_refine_ratio = torch.ones(
                    (b,),
                    device=nodes.device,
                    dtype=torch.float32,
                )
                adaptive_threshold = torch.zeros(
                    (b,),
                    device=nodes.device,
                    dtype=torch.float32,
                )

            delta_local, local_diag = self.local_refinement(
                x=nodes_with_pos,
                global_nodes=global_nodes,
                delta_global=delta_global,
                r=r,
                height=h,
                width=w,
                selected_indices=selected,
                label_prototypes=self.node_semantic_head.weight.detach(),
                selected_active_mask=selected_active_mask,
            )

            # Only truly active adaptive selections are reported as refined.
            sparse_mask = torch.zeros_like(r)
            active_values = selected_active_mask.to(dtype=r.dtype)
            sparse_mask.scatter_(1, selected, active_values)

            selected_scores = torch.gather(r, 1, selected)
            active_count = active_values.sum(dim=1).clamp_min(1.0)
            selected_need_mean = (
                selected_scores * active_values
            ).sum(dim=1) / active_count
        else:
            delta_local = torch.zeros_like(nodes_with_pos)
            sparse_mask = torch.zeros_like(r)
            selected = torch.empty(
                (b, 0),
                device=nodes.device,
                dtype=torch.long,
            )
            selected_active_mask = torch.empty(
                (b, 0),
                device=nodes.device,
                dtype=torch.bool,
            )
            dynamic_refine_ratio = torch.zeros(
                (b,),
                device=nodes.device,
                dtype=torch.float32,
            )
            adaptive_threshold = torch.zeros(
                (b,),
                device=nodes.device,
                dtype=torch.float32,
            )
            selected_need_mean = torch.zeros(
                (b,),
                device=nodes.device,
                dtype=torch.float32,
            )
            local_diag = {}

        # C3 Zero-Start Conservative Local Residual:
        # C1/C2 already decoupled routing demand R from the final correction
        # magnitude. C3 further protects the proven B6 representation by making
        # the local residual contribution EXACTLY zero at initialization.
        #
        # tanh keeps the learned strength bounded and signed:
        #   alpha_local > 0 : add the learned local correction
        #   alpha_local < 0 : optimizer can reverse an unhelpful correction
        #   alpha_local = 0 : exact B6-style global representation
        local_residual_strength = torch.tanh(
            self.local_residual_raw.float()
        )
        graph_nodes = (
            nodes_with_pos
            + delta_global
            + local_residual_strength * delta_local
        )
        graph_nodes = self.graph_fusion_norm(graph_nodes)
        graph_nodes = self.shared_ffn(graph_nodes)

        # B1: replace shared mean-only classification readout with label-specific
        # queries. The returned summary remains [B,D], so C/D/E and DRCF interfaces
        # stay unchanged.
        gnn_summary, gnn_aux_logits, readout_diag = self.global_readout(graph_nodes)
        gnn_feature = self.gnn_pool_norm(gnn_summary)

        # CNN relation representation is separate from the untouched 2048-D
        # baseline prediction path.
        # First-step isolation: ASRC relation feature still uses the original
        # Layer4 anchor. Only CNN logits are visually enhanced in C.
        cnn_relation_feature = self.cnn_relation_proj(pure_cnn_raw_feature)

        # ---------------- Fusion protocol ----------------
        # M3/M4: parameter-free 0.5/0.5 logit fusion.
        # M5/M6: original Chapter-4 ASRC.
        # M7/M8 (CA-ASRC): keep the empirically strong 0.5/0.5 consensus as the
        # decision anchor and let ASRC contribute only a bounded residual.
        # Crucially, the consensus anchor keeps full main-loss gradients to both
        # CNN and GNN, unlike the protected-detach adaptive base in original ASRC.
        fusion_mode = str(self.config.fusion_mode).lower()
        if fusion_mode == "fixed":
            correction_feature = torch.cat([cnn_relation_feature, gnn_feature], dim=-1)
            correction_logits = torch.zeros_like(cnn_logits)
            correction_alpha = torch.zeros(
                (b, 1), device=images.device, dtype=cnn_logits.dtype
            )
            correction_aux = {}
            graph_anchor_weight = torch.full_like(cnn_logits, 0.5)
            adaptive_base_logits = 0.5 * (cnn_logits + gnn_aux_logits)
            consensus_base_logits = adaptive_base_logits
            consensus_relation_residual = torch.zeros_like(cnn_logits)
            logits = adaptive_base_logits
        elif fusion_mode == "cp_asrc":
            # CP-ASRC directly generalizes fixed fusion.  There is no additive
            # relation-correction output competing with the consensus.  The
            # router only makes a bounded label-wise move along the natural
            # CNN/GNN disagreement direction.  M11 uses pure CNN logits here;
            # M12 recomputes the same unified fusion with SADRF-enhanced CNN
            # logits below, after the visual adapter is available.
            correction_feature = torch.cat([cnn_relation_feature, gnn_feature], dim=-1)
            correction_logits = torch.zeros_like(cnn_logits)
            correction_alpha = torch.zeros(
                (b, 1), device=images.device, dtype=cnn_logits.dtype
            )
            correction_aux = {}
            graph_anchor_weight = torch.full_like(cnn_logits, 0.5)
            if self.cp_fusion is None:
                raise RuntimeError("cp_asrc mode requires self.cp_fusion")
            logits, cp_aux = self.cp_fusion(
                cnn_relation_feature, gnn_feature, cnn_logits, gnn_aux_logits
            )
            correction_aux.update(cp_aux)
            consensus_base_logits = cp_aux["cp_consensus_logits"]
            adaptive_base_logits = consensus_base_logits
            consensus_relation_residual = logits - consensus_base_logits
        elif fusion_mode == "decoupled_asrc":
            # DCA-ASRC: preserve the empirically strong M3/M4 branch training
            # exactly, while ASRC learns only a detached residual correction.
            #
            # Base path (full gradient to CNN/GNN):
            #   z_cons = 0.5 z_C + 0.5 z_G
            # Correction path (no gradient to CNN/GNN):
            #   delta = R * s_label * tanh(h_rel)
            #   z_fusion_aux = stopgrad(z_cons) + delta
            # The trainer optimizes z_cons for the branches and z_fusion_aux only
            # for ASRC, with independent gradient clipping.
            correction_feature, correction_aux = self.relation_correction(
                cnn_relation_feature.detach(),
                gnn_feature.detach(),
            )
            correction_logits = self.correction_classifier(correction_feature)
            correction_alpha, demand_aux = self.label_correction_demand(
                cnn_logits=cnn_logits.detach(),
                gnn_logits=gnn_aux_logits.detach(),
            )
            correction_aux.update(demand_aux)
            graph_anchor_weight = demand_aux["label_graph_anchor_weight"]

            consensus_base_logits = 0.5 * (cnn_logits + gnn_aux_logits)
            adaptive_base_logits = consensus_base_logits

            dca_scale = (
                float(self.config.dca_residual_max)
                * torch.sigmoid(self.dca_scale_raw)
            ).view(1, -1)
            bounded_relation = torch.tanh(correction_logits)
            consensus_relation_residual = (
                correction_alpha * dca_scale * bounded_relation
            )
            # Forward final prediction.  Training isolation is enforced by
            # training_base_logits / fusion_aux_logits below.
            logits = consensus_base_logits + consensus_relation_residual
            correction_aux["dca_scale_mean"] = dca_scale.detach().mean(dim=-1)
            correction_aux["dca_scale_max"] = dca_scale.detach().amax(dim=-1)
        elif fusion_mode == "consensus_asrc":
            # Relation representation keeps the Chapter-4 one-way correction
            # design: CNN reference is detached and GNN correction-path gradient
            # is weakly coupled by gnn_correction_grad_scale.
            correction_feature, correction_aux = self.relation_correction(
                cnn_relation_feature,
                gnn_feature,
            )
            correction_logits = self.correction_classifier(correction_feature)

            # Reuse the Chapter-4 label-wise correction-demand evidence, but do
            # NOT use its confidence weight to replace the consensus anchor.
            correction_alpha, demand_aux = self.label_correction_demand(
                cnn_logits=cnn_logits,
                gnn_logits=gnn_aux_logits,
            )
            correction_aux.update(demand_aux)
            graph_anchor_weight = demand_aux["label_graph_anchor_weight"]

            # Strong consensus anchor. This is exactly M3/M4 at initialization
            # because correction_classifier is zero-initialized. Full gradient
            # from the main loss reaches both branch logits through this path.
            consensus_base_logits = 0.5 * (cnn_logits + gnn_aux_logits)
            adaptive_base_logits = consensus_base_logits

            # ASRC is demoted from 'base constructor' to 'relation residual'.
            # tanh bounds each label correction to [-1,1] before the structural
            # demand gate, preventing relation correction from overwhelming the
            # already strong consensus ranking.
            bounded_relation = torch.tanh(correction_logits)
            consensus_relation_residual = correction_alpha * bounded_relation
            logits = consensus_base_logits + consensus_relation_residual
        elif self.config.use_drcf:
            # Keep the proven E2.1 one-way relation-correction representation.
            correction_feature, correction_aux = self.relation_correction(
                cnn_relation_feature,
                gnn_feature,
            )
            correction_logits = self.correction_classifier(correction_feature)

            # V14 keeps the Chapter-4 relation policy untouched.  SADRF does
            # not alter branch-confidence anchoring or correction demand here.
            correction_alpha, demand_aux = self.label_correction_demand(
                cnn_logits=cnn_logits,
                gnn_logits=gnn_aux_logits,
            )
            correction_aux.update(demand_aux)
            graph_anchor_weight = demand_aux["label_graph_anchor_weight"]

            # E4.1 protected adaptive-base path.
            #
            # E4 showed that the adaptive anchor is useful for Final prediction,
            # but allowing Final-loss gradients from this direct logit-mixing
            # path to reach the GNN can damage the GNN's standalone classifier.
            #
            # Therefore:
            #   forward value:  exactly gnn_aux_logits
            #   backward into GNN from adaptive-base path: 0
            #
            # The GNN still receives:
            #   (1) its full auxiliary classification loss, and
            #   (2) the weak correction-path gradient (default 0.25) through
            #       GraphRelationCorrection.
            gnn_logits_for_base = gnn_aux_logits.detach()

            # Label-wise adaptive branch anchor:
            #   zBase_i = (1-lambdaG_i) zCNN_i + lambdaG_i zGNN_i
            adaptive_base_logits = (
                (1.0 - graph_anchor_weight) * cnn_logits
                + graph_anchor_weight * gnn_logits_for_base
            )
            consensus_base_logits = 0.5 * (cnn_logits + gnn_aux_logits)

            # E4:
            #   zFinal_i = zBase_i + R_i^F * Delta z_i^relation
            consensus_relation_residual = correction_alpha * correction_logits
            logits = (
                adaptive_base_logits
                + consensus_relation_residual
            )
        else:
            # B/C/D baseline fusion is unchanged.
            correction_feature = self.simple_fusion(
                torch.cat([cnn_relation_feature, gnn_feature], dim=-1)
            )
            correction_logits = self.correction_classifier(correction_feature)
            correction_alpha = torch.ones(
                (b, 1),
                device=images.device,
                dtype=correction_logits.dtype,
            )
            correction_aux = {}
            adaptive_base_logits = cnn_logits
            consensus_base_logits = cnn_logits
            consensus_relation_residual = correction_alpha * correction_logits
            logits = cnn_logits + consensus_relation_residual

        # ``logits`` above is the complete original SGRC prediction.  V14
        # computes the SADRF correction only *after* this path is finished.
        # The trainer uses sgrc_logits for the original SGRC main loss, while
        # detached SGRC/CNN anchors are used to train the visual residual.
        sgrc_logits = logits
        # Default behavior for legacy modes. DCA-ASRC overrides the training base
        # with the fixed consensus so the branch optimization is identical to M3/M4.
        training_base_logits = sgrc_logits
        fusion_aux_logits = None
        enhanced_consensus_logits = consensus_base_logits
        if fusion_mode == "decoupled_asrc":
            training_base_logits = consensus_base_logits
            fusion_aux_logits = (
                consensus_base_logits.detach() + consensus_relation_residual
            )
        elif fusion_mode == "cp_asrc":
            # One end-to-end fusion objective; no parallel residual objective.
            training_base_logits = sgrc_logits
        sadrf_final_aux_logits = sgrc_logits.detach()
        if self.lite_sadrf is not None:
            enhanced_cnn_logits, cnn_visual_delta, final_visual_delta, lite_aux = self.lite_sadrf(
                stage3=layer3_map,
                layer4_feature=pure_cnn_raw_feature,
                pure_cnn_logits=pure_cnn_logits,
                sgrc_logits=sgrc_logits,
            )
            if fusion_mode == "fixed":
                # M4: keep exactly the same GNN and fixed fusion as M3.
                logits = 0.5 * (enhanced_cnn_logits + gnn_aux_logits)
                sadrf_final_aux_logits = 0.5 * (
                    enhanced_cnn_logits + gnn_aux_logits.detach()
                )
            elif fusion_mode == "cp_asrc":
                # M12 uses SADRF as the actual CNN branch inside the *same*
                # adaptive fusion equation.  Reconstruct the enhanced logits
                # without detaching pure_cnn_logits so the final loss jointly
                # trains CNN, SADRF and GNN instead of selecting between two
                # separately-computed mechanisms.
                joint_enhanced_cnn = pure_cnn_logits + cnn_visual_delta.to(pure_cnn_logits.dtype)
                logits, cp_aux = self.cp_fusion(
                    cnn_relation_feature, gnn_feature, joint_enhanced_cnn, gnn_aux_logits
                )
                correction_aux.update(cp_aux)
                consensus_base_logits = cp_aux["cp_consensus_logits"]
                adaptive_base_logits = consensus_base_logits
                enhanced_consensus_logits = consensus_base_logits
                consensus_relation_residual = logits - consensus_base_logits
                sgrc_logits = logits
                training_base_logits = logits
                enhanced_cnn_logits = joint_enhanced_cnn
                # No second "final residual" objective in CP-ASRC.  The CNN
                # sidecar is supervised below and participates directly in the
                # unified main fusion objective.
                final_visual_delta = 0.5 * cnn_visual_delta.to(logits.dtype)
            elif fusion_mode == "decoupled_asrc":
                # M10: use exactly the M4 validated CNN enhancement in the
                # consensus anchor, while keeping both the SADRF and ASRC losses
                # detached from the base CNN/GNN training trajectory.
                enhanced_consensus_logits = 0.5 * (
                    enhanced_cnn_logits + gnn_aux_logits
                )
                logits = enhanced_consensus_logits + consensus_relation_residual
                sadrf_final_aux_logits = 0.5 * (
                    enhanced_cnn_logits + gnn_aux_logits.detach()
                )
                fusion_aux_logits = (
                    enhanced_consensus_logits.detach()
                    + consensus_relation_residual
                )
                # The V15 final head is not used in DCA; only the validated
                # CNN enhancement enters the consensus.
                final_visual_delta = 0.5 * cnn_visual_delta.to(sgrc_logits.dtype)
            elif fusion_mode == "consensus_asrc":
                # M8: preserve the M7 CA-ASRC core and inject the SAME CNN
                # enhancement already validated by M2/M4. Because the consensus
                # anchor gives CNN weight 0.5, replacing pure CNN by enhanced CNN
                # is exactly equivalent to adding 0.5 * cnn_visual_delta.
                visual_consensus_delta = 0.5 * cnn_visual_delta.to(sgrc_logits.dtype)
                logits = sgrc_logits + visual_consensus_delta
                sadrf_final_aux_logits = (
                    sgrc_logits.detach() + 0.5 * cnn_visual_delta
                )
                # For regularization/diagnostics use the actual final visual
                # contribution rather than the unused V15 second head.
                final_visual_delta = visual_consensus_delta
            else:
                # M6: V15-style visual residual on top of the complete SGRC path.
                logits = sgrc_logits + final_visual_delta.to(sgrc_logits.dtype)
                sadrf_final_aux_logits = sgrc_logits.detach() + final_visual_delta
        else:
            logits = sgrc_logits

        # Preserve the original SGRC CNN auxiliary objective.  A separate SADRF
        # auxiliary prediction teaches only the detached visual sidecar.
        cnn_aux_logits = pure_cnn_logits
        node_aux_logits = node_logits.max(dim=1).values

        output: Dict[str, Tensor] = {
            "logits": logits,
            "sgrc_logits": sgrc_logits,
            "cnn_aux_logits": cnn_aux_logits,
            "gnn_aux_logits": gnn_aux_logits,
            "node_aux_logits": node_aux_logits,
            "cnn_logits": enhanced_cnn_logits,
            "pure_cnn_logits": pure_cnn_logits,
            "training_base_logits": training_base_logits,
            "adaptive_base_logits": adaptive_base_logits,
            "consensus_base_logits": consensus_base_logits,
            "consensus_relation_residual": consensus_relation_residual,
            "consensus_correction_abs_mean": consensus_relation_residual.detach().abs().mean(dim=-1),
            "enhanced_consensus_logits": enhanced_consensus_logits,
            "correction_logits": correction_logits,
            "correction_alpha": correction_alpha,
            "adaptive_base_gnn_grad_scale": torch.zeros(
                (),
                device=logits.device,
                dtype=logits.dtype,
            ),
            "refinement_score": r,
            "refinement_score_cmrde": r_cmrde,
            "local_residual_strength": local_residual_strength.detach(),
            "semantic_relevance": semantic_relevance,
            "refinement_mask": sparse_mask,
            "selected_indices": selected,
            "selected_active_mask": selected_active_mask,
            "dynamic_refine_ratio": dynamic_refine_ratio,
            "adaptive_refine_threshold": adaptive_threshold,
            "selected_need_mean": selected_need_mean,
            "u_rel": u_rel,
            "u_cls": u_cls,
            "u_local": u_local,
            "evidence_weights": evidence_weights,
            "cnn_feature": cnn_relation_feature,
            "cnn_raw_feature": cnn_raw_feature,
            "pure_cnn_raw_feature": pure_cnn_raw_feature,
            "gnn_feature": gnn_feature,
            # Trainer only requires a finite tensor under this diagnostic key.
            "fused_feature": correction_feature,
            "label_readout_entropy": readout_diag["label_readout_entropy"],
            "label_readout_peak": readout_diag["label_readout_peak"],
            "prediction_aggregation_strength": readout_diag[
                "prediction_aggregation_strength"
            ],
            "label_confidence_mean": readout_diag["label_confidence_mean"],
            "label_confidence_entropy": readout_diag[
                "label_confidence_entropy"
            ],
            "relation_calibration_strength": calibration_diag[
                "relation_calibration_strength"
            ],
            "relation_residual_gate_mean": calibration_diag[
                "relation_residual_gate_mean"
            ],
            "relation_residual_gate_min": calibration_diag[
                "relation_residual_gate_min"
            ],
            "relation_residual_gate_max": calibration_diag[
                "relation_residual_gate_max"
            ],
        }

        if self.lite_sadrf is not None:
            # The CNN-side SADRF objective remains useful in every mode.  CP-ASRC
            # deliberately has no separate final-side residual target because the
            # enhanced CNN is already fused end-to-end by the unified main loss.
            output["sadrf_cnn_aux_logits"] = pure_cnn_logits.detach() + cnn_visual_delta
            if fusion_mode != "cp_asrc":
                output["sadrf_final_aux_logits"] = sadrf_final_aux_logits
            output["sadrf_cnn_delta"] = cnn_visual_delta
            output["sadrf_final_delta"] = final_visual_delta

        if fusion_aux_logits is not None:
            output["fusion_aux_logits"] = fusion_aux_logits
        for _cp_key in (
            "cp_beta", "cp_beta_abs_mean", "cp_beta_min", "cp_beta_max",
            "cp_cnn_weight_mean", "cp_gnn_weight_mean", "cp_probability_disagreement",
        ):
            if _cp_key in correction_aux:
                output[_cp_key] = correction_aux[_cp_key]
        if self.dca_scale_raw is not None:
            _dca_scale = (
                float(self.config.dca_residual_max)
                * torch.sigmoid(self.dca_scale_raw)
            )
            output["dca_scale_mean"] = _dca_scale.detach().mean()
            output["dca_scale_max"] = _dca_scale.detach().max()

        if "relation_gap_norm" in correction_aux:
            output["relation_gap_norm"] = correction_aux["relation_gap_norm"]
        for _key in (
            "label_correction_demand",
            "label_disagreement",
            "label_graph_support",
            "label_graph_anchor_weight",
            "label_cnn_uncertainty",
            "label_gnn_uncertainty",
            "label_demand_learned_gate",
        ):
            if _key in correction_aux:
                output[_key] = correction_aux[_key]

        if "relation_concentration" in attn_diag:
            output["relation_concentration"] = attn_diag[
                "relation_concentration"
            ]
        if "ls_scna_gate" in local_diag:
            output["ls_scna_gate"] = local_diag["ls_scna_gate"]
        if "ls_scna_base_gate" in local_diag:
            output["ls_scna_base_gate"] = local_diag["ls_scna_base_gate"]
        if "ls_scna_residual_strength" in local_diag:
            output["ls_scna_residual_strength"] = local_diag[
                "ls_scna_residual_strength"
            ]
        if "ls_scna_ambiguity" in local_diag:
            output["ls_scna_ambiguity"] = local_diag[
                "ls_scna_ambiguity"
            ]
        if "ls_scna_reliable_ambiguity" in local_diag:
            output["ls_scna_reliable_ambiguity"] = local_diag[
                "ls_scna_reliable_ambiguity"
            ]
        if "ls_scna_bell_reliability" in local_diag:
            output["ls_scna_bell_reliability"] = local_diag[
                "ls_scna_bell_reliability"
            ]
        if "ls_scna_low_margin_recovery" in local_diag:
            output["ls_scna_low_margin_recovery"] = local_diag[
                "ls_scna_low_margin_recovery"
            ]
        if "ls_scna_top12_margin" in local_diag:
            output["ls_scna_top12_margin"] = local_diag[
                "ls_scna_top12_margin"
            ]
        if "ls_scna_assignment_entropy" in local_diag:
            output["ls_scna_assignment_entropy"] = local_diag[
                "ls_scna_assignment_entropy"
            ]
        if "mean_edge_gate" in local_diag:
            output["mean_edge_gate"] = local_diag["mean_edge_gate"]

        for _key, _value in lite_aux.items():
            output[_key] = _value

        return output

    def parameter_groups(self, backbone_lr: float, head_lr: float):
        """Optimizer groups: lower LR for pretrained CNN, higher LR for new modules."""
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        new_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": new_params, "lr": head_lr},
        ]


def build_model(
    num_classes: int,
    embed_dim: int = 256,
    num_heads: int = 4,
    favor_features: int = 64,
    favor_orthogonal_scaling: int = 0,
    refine_ratio: float = 0.25,
    min_refine_nodes: int = 4,
    local_kernel_size: int = 3,
    ls_scna_temperature: float = 0.2,
    dropout: float = 0.1,
    pretrained_backbone: bool = True,
    backbone_weights: str = "",
    use_cmrde: bool = True,
    use_local_refinement: bool = True,
    use_drcf: bool = True,
    use_ls_scna: bool = True,
    gnn_correction_grad_scale: float = 0.25,
    use_lite_sadrf: bool = False,
    lite_sadrf_hidden: int = 48,
    lite_sadrf_delta_max: float = 0.30,
    fusion_mode: str = "full",
    dca_residual_max: float = 0.25,
    dca_scale_init: float = 0.05,
    cp_beta_max: float = 0.20,
    cp_hidden_dim: int = 64,
) -> GLRDRCFNet:
    config = GLRDRCFConfig(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        favor_features=favor_features,
        favor_orthogonal_scaling=favor_orthogonal_scaling,
        refine_ratio=refine_ratio,
        min_refine_nodes=min_refine_nodes,
        local_kernel_size=local_kernel_size,
        ls_scna_temperature=ls_scna_temperature,
        dropout=dropout,
        pretrained_backbone=pretrained_backbone,
        backbone_weights=backbone_weights,
        use_cmrde=use_cmrde,
        use_local_refinement=use_local_refinement,
        use_drcf=use_drcf,
        use_ls_scna=use_ls_scna,
        gnn_correction_grad_scale=gnn_correction_grad_scale,
        use_lite_sadrf=use_lite_sadrf,
        lite_sadrf_hidden=lite_sadrf_hidden,
        lite_sadrf_delta_max=lite_sadrf_delta_max,
        fusion_mode=fusion_mode,
        dca_residual_max=dca_residual_max,
        dca_scale_init=dca_scale_init,
        cp_beta_max=cp_beta_max,
        cp_hidden_dim=cp_hidden_dim,
    )
    return GLRDRCFNet(config)


# -----------------------------------------------------------------------------
# Chapter-5 M1--M6 controlled validation models
# -----------------------------------------------------------------------------

class ResNet101MultiLabelBaseline(nn.Module):
    """M1: pure ResNet-101 multi-label CNN baseline under the common protocol."""
    def __init__(self, num_classes: int, pretrained_backbone: bool = True, backbone_weights: str = "") -> None:
        super().__init__()
        self.backbone = ResNet101FeatureExtractor(pretrained=pretrained_backbone, local_weights=backbone_weights)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.out_channels, int(num_classes))

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        _, _, l4 = self.backbone.forward_hierarchy(images)
        feat = self.pool(l4).flatten(1)
        logits = self.classifier(feat)
        return {
            "logits": logits,
            "training_base_logits": logits,
            "cnn_logits": logits,
            "pure_cnn_logits": logits,
            "cnn_raw_feature": feat,
        }

    def parameter_groups(self, backbone_lr: float, head_lr: float):
        bp = list(self.backbone.parameters())
        ids = {id(x) for x in bp}
        hp = [x for x in self.parameters() if id(x) not in ids]
        return [{"params": bp, "lr": backbone_lr}, {"params": hp, "lr": head_lr}]


class AdaptedSADRFResNet101Baseline(nn.Module):
    """M2: ResNet-101 + the exact V15 Lite-SADRF visual adaptation, CNN only.

    The ordinary ResNet classifier is optimized by the main BCE.  The Lite-SADRF
    branch reads detached Stage3/Layer4 features and is optimized by its own BCE,
    so the experiment measures whether the Chapter-3 visual idea transfers to
    the multi-label task without help from any GNN/fusion component.
    """
    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        backbone_weights: str = "",
        lite_sadrf_hidden: int = 48,
        lite_sadrf_delta_max: float = 0.30,
    ) -> None:
        super().__init__()
        self.backbone = ResNet101FeatureExtractor(pretrained=pretrained_backbone, local_weights=backbone_weights)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbone.out_channels, int(num_classes))
        # Keep the attribute name used by the trainer's independent clipping.
        self.lite_sadrf = LiteSADRFSemanticResidualAdapter(
            num_classes=int(num_classes), hidden_channels=int(lite_sadrf_hidden), delta_max=float(lite_sadrf_delta_max)
        )

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        _, l3, l4 = self.backbone.forward_hierarchy(images)
        feat = self.pool(l4).flatten(1)
        pure = self.classifier(feat)
        enhanced, cnn_delta, _unused_final_delta, aux = self.lite_sadrf(
            stage3=l3,
            layer4_feature=feat,
            pure_cnn_logits=pure,
            sgrc_logits=pure,
        )
        out: Dict[str, Tensor] = {
            "logits": enhanced,
            "training_base_logits": pure,
            "cnn_logits": enhanced,
            "pure_cnn_logits": pure,
            # Give the transferred visual branch a full-strength detached target.
            "sadrf_final_aux_logits": enhanced,
            "sadrf_cnn_delta": cnn_delta,
            "cnn_raw_feature": feat,
        }
        out.update(aux)
        return out

    def parameter_groups(self, backbone_lr: float, head_lr: float):
        bp = list(self.backbone.parameters())
        ids = {id(x) for x in bp}
        hp = [x for x in self.parameters() if id(x) not in ids]
        return [{"params": bp, "lr": backbone_lr}, {"params": hp, "lr": head_lr}]


def build_validation_model(
    experiment: str,
    num_classes: int,
    embed_dim: int = 256,
    num_heads: int = 4,
    favor_features: int = 64,
    favor_orthogonal_scaling: int = 0,
    refine_ratio: float = 0.15,
    min_refine_nodes: int = 4,
    local_kernel_size: int = 3,
    ls_scna_temperature: float = 0.2,
    dropout: float = 0.1,
    pretrained_backbone: bool = True,
    backbone_weights: str = "",
    gnn_correction_grad_scale: float = 0.25,
    lite_sadrf_hidden: int = 48,
    lite_sadrf_delta_max: float = 0.30,
    dca_residual_max: float = 0.25,
    dca_scale_init: float = 0.05,
    cp_beta_max: float = 0.20,
    cp_hidden_dim: int = 64,
) -> nn.Module:
    """Build thesis-validation models M1--M12, including CP-ASRC M11/M12."""
    exp = str(experiment).upper().strip()
    if exp == "M1":
        return ResNet101MultiLabelBaseline(num_classes, pretrained_backbone, backbone_weights)
    if exp == "M2":
        return AdaptedSADRFResNet101Baseline(
            num_classes, pretrained_backbone, backbone_weights,
            lite_sadrf_hidden, lite_sadrf_delta_max,
        )
    if exp not in {"M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"}:
        raise ValueError(f"Unknown validation experiment: {experiment}")

    fixed = exp in {"M3", "M4"}
    consensus_asrc = exp in {"M7", "M8"}
    decoupled_asrc = exp in {"M9", "M10"}
    cp_asrc = exp in {"M11", "M12"}
    enhanced = exp in {"M4", "M6", "M8", "M10", "M12"}
    return build_model(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        favor_features=favor_features,
        favor_orthogonal_scaling=favor_orthogonal_scaling,
        refine_ratio=refine_ratio,
        min_refine_nodes=min_refine_nodes,
        local_kernel_size=local_kernel_size,
        ls_scna_temperature=ls_scna_temperature,
        dropout=dropout,
        pretrained_backbone=pretrained_backbone,
        backbone_weights=backbone_weights,
        # The GNN is kept identical in M3--M6; only fusion and CNN enhancement change.
        use_cmrde=True,
        use_local_refinement=True,
        use_drcf=not fixed,
        use_ls_scna=True,
        gnn_correction_grad_scale=gnn_correction_grad_scale,
        use_lite_sadrf=enhanced,
        lite_sadrf_hidden=lite_sadrf_hidden,
        lite_sadrf_delta_max=lite_sadrf_delta_max,
        fusion_mode=("fixed" if fixed else ("cp_asrc" if cp_asrc else ("decoupled_asrc" if decoupled_asrc else ("consensus_asrc" if consensus_asrc else "full")))),
        dca_residual_max=dca_residual_max,
        dca_scale_init=dca_scale_init,
        cp_beta_max=cp_beta_max,
        cp_hidden_dim=cp_hidden_dim,
    )
