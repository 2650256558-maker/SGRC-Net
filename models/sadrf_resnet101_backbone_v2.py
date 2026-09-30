"""
SADRF-enhanced ResNet-101 backbone for the Chapter-5 SGRC-Net integration.

Design principle
----------------
This is NOT a second complete SADRFNet branch. It keeps the full ImageNet-
pretrained ResNet-101 as the visual backbone and ports only the three CNN-side
innovations validated by SADRFNet:

1) Stage-2/Stage-3 Dilated Reparameterized Context Enhancement (DRCE)
2) scene-adaptive local/global complementary fusion at those stages
3) progressive multi-stage fusion + semantic-anchored residual fusion

The output interfaces are deliberately kept compatible with SGRC-Net:
- zero-start context-adapted Layer3 map: [B, 1024, H/16, W/16] -> graph node source
- pure Layer4 map:     [B, 2048, H/32, W/32] -> unchanged SGRC semantic anchor
- enhanced CNN vector:[B, 2048]              -> CNN logits/semantic anchor/ASRC

For ResNet-101 we preserve the ABSOLUTE SADRF context-path depths (1 DRB unit in
Stage 2 and 2 units in Stage 3) rather than scaling the ResNet-50 depth ratios.
This avoids turning ResNet-101 Stage 3 (23 bottlenecks) into an unnecessarily
large 7-8 unit DRB branch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.models import ResNet101_Weights
except Exception:  # pragma: no cover
    torchvision = None
    ResNet101_Weights = None


def _make_divisible(v: int, divisor: int = 8) -> int:
    return int((v + divisor - 1) // divisor * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act_layer: Optional[type[nn.Module]] = nn.GELU,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act_layer is not None:
            layers.append(act_layer())
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(_make_divisible(channels // reduction), 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(self.avg_pool(x))


class ConvFFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = _make_divisible(int(channels * expansion))
        self.norm = nn.BatchNorm2d(channels)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.bn(self.dwconv(x)))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return identity + x


class _DWConvBN(nn.Sequential):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        effective_kernel = dilation * (kernel_size - 1) + 1
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=effective_kernel // 2,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.kernel_size = kernel_size
        self.dilation = dilation


class DilatedReparamSpatialBlock(nn.Module):
    """Training: large DW branch + dilated 3x3 DW branches; deploy: one DW conv."""

    def __init__(
        self,
        channels: int,
        large_kernel_size: int,
        dilations: Sequence[int],
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if large_kernel_size % 2 == 0:
            raise ValueError("large_kernel_size must be odd")
        self.channels = int(channels)
        self.large_kernel_size = int(large_kernel_size)
        self.deploy = bool(deploy)

        if deploy:
            self.deploy_dw = nn.Conv2d(
                channels,
                channels,
                large_kernel_size,
                padding=large_kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.large_branch = _DWConvBN(channels, large_kernel_size, dilation=1)
            branches = []
            for d in dilations:
                effective = int(d) * 2 + 1
                if effective > large_kernel_size:
                    raise ValueError(
                        f"dilation={d} gives effective kernel {effective}, "
                        f"larger than large_kernel_size={large_kernel_size}"
                    )
                branches.append(_DWConvBN(channels, 3, dilation=int(d)))
            self.dilated_branches = nn.ModuleList(branches)

        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            out = self.deploy_dw(x)
        else:
            out = self.large_branch(x)
            for branch in self.dilated_branches:
                out = out + branch(x)
        return self.pw(out)

    @staticmethod
    def _fuse_conv_bn(branch: _DWConvBN) -> Tuple[Tensor, Tensor]:
        conv, bn = branch[0], branch[1]
        weight = conv.weight
        bias = torch.zeros(weight.size(0), device=weight.device, dtype=weight.dtype)
        if conv.bias is not None:
            bias = conv.bias
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_weight = weight * scale
        fused_bias = bn.bias + (bias - bn.running_mean) * bn.weight / std
        return fused_weight, fused_bias

    @staticmethod
    def _dilate_kernel(kernel: Tensor, dilation: int) -> Tensor:
        if dilation == 1:
            return kernel
        c, one, k, _ = kernel.shape
        effective = dilation * (k - 1) + 1
        out = kernel.new_zeros((c, one, effective, effective))
        out[:, :, ::dilation, ::dilation] = kernel
        return out

    @staticmethod
    def _pad_kernel(kernel: Tensor, target: int) -> Tensor:
        current = kernel.size(-1)
        if current == target:
            return kernel
        total = target - current
        if total < 0:
            raise ValueError("kernel larger than target")
        left = total // 2
        right = total - left
        return F.pad(kernel, [left, right, left, right])

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.deploy_dw.weight, self.deploy_dw.bias
        kernel, bias = self._fuse_conv_bn(self.large_branch)
        kernel = self._pad_kernel(kernel, self.large_kernel_size)
        for branch in self.dilated_branches:
            k, b = self._fuse_conv_bn(branch)
            k = self._dilate_kernel(k, branch.dilation)
            k = self._pad_kernel(k, self.large_kernel_size)
            kernel = kernel + k
            bias = bias + b
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.deploy_dw = nn.Conv2d(
            self.channels,
            self.channels,
            self.large_kernel_size,
            padding=self.large_kernel_size // 2,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        self.deploy_dw.weight.copy_(kernel)
        self.deploy_dw.bias.copy_(bias)
        del self.large_branch
        del self.dilated_branches
        self.deploy = True


class GlobalDRBContextUnit(nn.Module):
    def __init__(
        self,
        channels: int,
        large_kernel_size: int,
        dilations: Sequence[int],
        ffn_expansion: float = 1.0,
        deploy: bool = False,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(channels)
        self.spatial = DilatedReparamSpatialBlock(
            channels, large_kernel_size, dilations, deploy=deploy
        )
        self.act = nn.GELU()
        self.se = SqueezeExcitation(channels, reduction=16)
        self.layer_scale = nn.Parameter(torch.full((channels,), float(layer_scale_init)))
        self.ffn = ConvFFN(channels, expansion=max(float(ffn_expansion), 1.0))

    def forward(self, x: Tensor) -> Tensor:
        y = self.act(self.spatial(self.pre_norm(x)))
        y = self.se(y)
        x = x + self.layer_scale.view(1, -1, 1, 1) * y
        return self.ffn(x)


class GlobalDRBContextPath(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        depth: int,
        large_kernel_size: int,
        dilations: Sequence[int],
        ffn_expansion: float = 1.0,
        global_residual_init: float = 0.05,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.reduce = ConvBNAct(channels, hidden_channels, 1, act_layer=nn.GELU)
        self.blocks = nn.Sequential(
            *[
                GlobalDRBContextUnit(
                    hidden_channels,
                    large_kernel_size,
                    dilations,
                    ffn_expansion=ffn_expansion,
                    deploy=deploy,
                )
                for _ in range(depth)
            ]
        )
        self.expand = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_residual_scale = nn.Parameter(torch.tensor(float(global_residual_init)))

    def forward(self, local_feature: Tensor) -> Tensor:
        context = self.expand(self.blocks(self.reduce(local_feature)))
        return local_feature + torch.tanh(self.global_residual_scale) * context


class LocalGlobalStageFusion(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, local_prior: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < local_prior < 1.0:
            raise ValueError("local_prior must be in (0,1)")
        hidden = max(channels // reduction, 32)
        self.channels = int(channels)
        self.local_prior = float(local_prior)
        self.router = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )
        self.difference_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_act = nn.ReLU(inplace=False)
        self.last_branch_balance: Optional[Tensor] = None
        self.reset_fusion_parameters()

    def reset_fusion_parameters(self) -> None:
        final = self.router[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias[: self.channels].fill_(float(torch.log(torch.tensor(self.local_prior))))
            final.bias[self.channels :].fill_(float(torch.log(torch.tensor(1.0 - self.local_prior))))
        final_bn = self.difference_proj[-1]
        nn.init.zeros_(final_bn.weight)
        nn.init.zeros_(final_bn.bias)

    def forward(self, local_feature: Tensor, global_feature: Tensor) -> Tuple[Tensor, Tensor]:
        la = F.adaptive_avg_pool2d(local_feature, 1)
        lm = F.adaptive_max_pool2d(local_feature, 1)
        ga = F.adaptive_avg_pool2d(global_feature, 1)
        gm = F.adaptive_max_pool2d(global_feature, 1)
        logits = self.router(torch.cat([la, lm, ga, gm], dim=1))
        b = logits.size(0)
        weights = torch.softmax(logits.view(b, 2, self.channels, 1, 1), dim=1)
        mixture = weights[:, 0] * local_feature + weights[:, 1] * global_feature
        complementary = self.difference_proj(torch.abs(local_feature - global_feature))
        fused = self.out_act(mixture + complementary)
        balance = weights.mean(dim=2).flatten(1)
        self.last_branch_balance = balance.detach().mean(dim=0)
        return fused, balance


class StageResidualRefine(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return self.act(x + self.refine(x))


class DownsampleForFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = ConvBNAct(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        x = self.down(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class PairFusionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )
        self.concat_proj = ConvBNAct(channels * 2, channels, 1)
        self.refine = StageResidualRefine(channels, channels)

    def forward(self, x_low: Tensor, x_high: Tensor) -> Tuple[Tensor, Tensor]:
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_low = F.interpolate(x_low, size=x_high.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([x_low, x_high], dim=1)
        weights = torch.softmax(self.gate(cat), dim=1)
        weighted = weights[:, 0:1] * x_low + weights[:, 1:2] * x_high
        fused = self.refine(weighted + self.concat_proj(cat))
        return fused, weights.flatten(1)


class ProgressiveStageFusion(nn.Module):
    def __init__(self, in_channels_list: Sequence[int], fusion_channels: int = 256) -> None:
        super().__init__()
        self.residuals = nn.ModuleList(
            [StageResidualRefine(c, fusion_channels) for c in in_channels_list]
        )
        self.down12 = DownsampleForFusion(fusion_channels)
        self.down23 = DownsampleForFusion(fusion_channels)
        self.down34 = DownsampleForFusion(fusion_channels)
        self.fuse2 = PairFusionBlock(fusion_channels)
        self.fuse3 = PairFusionBlock(fusion_channels)
        self.fuse4 = PairFusionBlock(fusion_channels)

    def forward(self, features: Sequence[Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        x1, x2, x3, x4 = features
        r1, r2, r3, r4 = [m(x) for m, x in zip(self.residuals, features)]
        f2, w2 = self.fuse2(self.down12(r1, r2.shape[-2:]), r2)
        f3, w3 = self.fuse3(self.down23(f2, r3.shape[-2:]), r3)
        f4, w4 = self.fuse4(self.down34(f3, r4.shape[-2:]), r4)
        return f4, {
            "stage2_pair_weights": w2,
            "stage3_pair_weights": w3,
            "stage4_pair_weights": w4,
        }


class SceneAdaptiveResidualFusion(nn.Module):
    """SGRC-safe semantic-anchored residual fusion.

    This keeps the SADRF idea of using the final ResNet stage as a stable
    semantic anchor and injecting progressive multi-stage detail adaptively,
    but changes the injection to an EXACT zero-start residual:

        z_cnn = GAP(x4) + tanh(alpha_raw) * gate * Delta(f4)

    At initialization ``alpha_raw=0``, so ``z_cnn == GAP(x4)`` exactly. This
    makes the Chapter-5 model begin from the original SGRC-Net CNN feature
    distribution instead of replacing it with a normalized SADRF embedding.
    No dropout is used inside this side path, so it also does not perturb the
    RNG sequence used by SGRC-Net's downstream stochastic modules.
    """

    def __init__(
        self,
        original_channels: int = 2048,
        fusion_channels: int = 256,
        embed_dim: int = 2048,
        dropout: float = 0.0,  # retained for API compatibility; intentionally unused
        init_residual_gate: float = -2.0,
    ) -> None:
        super().__init__()
        if original_channels != embed_dim:
            raise ValueError(
                "SGRC-safe semantic anchor expects original_channels == embed_dim "
                "so the zero-start path can exactly preserve GAP(Stage4)."
            )
        self.embed_dim = int(embed_dim)

        # Only the residual branch is normalized/projected. The anchor itself
        # remains the exact raw GAP feature used by original SGRC-Net.
        self.anchor_norm = nn.LayerNorm(embed_dim)
        self.fused_proj = nn.Sequential(
            nn.Linear(fusion_channels * 2, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        hidden = max(embed_dim // 4, 128)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, float(init_residual_gate))

        detail_hidden = max(embed_dim // 2, 256)
        self.detail_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, detail_hidden, bias=False),
            nn.GELU(),
            nn.Linear(detail_hidden, embed_dim, bias=False),
        )

        # Signed zero-start shared scale. This is the key non-interference
        # protection: the enhanced CNN path starts exactly as original SGRC.
        self.residual_raw = nn.Parameter(torch.zeros(()))

    def forward(self, x4: Tensor, f4: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z_anchor = F.adaptive_avg_pool2d(x4, 1).flatten(1)
        z_anchor_n = self.anchor_norm(z_anchor)

        avg = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        mx = F.adaptive_max_pool2d(f4, 1).flatten(1)
        z_fused = self.fused_proj(torch.cat([avg, mx], dim=1))

        gate_input = torch.cat(
            [
                z_anchor_n,
                z_fused,
                torch.abs(z_anchor_n - z_fused),
                z_anchor_n * z_fused,
            ],
            dim=1,
        )
        base_gate = torch.sigmoid(self.gate_mlp(gate_input))
        detail = self.detail_proj(z_fused - z_anchor_n)
        residual_strength = torch.tanh(self.residual_raw)
        z = z_anchor + residual_strength * base_gate * detail
        return z, base_gate, residual_strength


def _strip_state_dict_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    cleaned: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _build_resnet101(pretrained: bool = True, local_weights: str = "") -> nn.Module:
    if torchvision is None:
        raise ImportError("torchvision is required for ResNet-101")
    try:
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained and not local_weights else None
        model = torchvision.models.resnet101(weights=weights)
    except Exception:
        model = torchvision.models.resnet101(pretrained=pretrained and not local_weights)

    if local_weights:
        ckpt = torch.load(local_weights, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported backbone checkpoint: {local_weights}")
        missing, unexpected = model.load_state_dict(_strip_state_dict_prefix(ckpt), strict=False)
        print(f"[SADRF-R101] Loaded local ResNet-101 weights: {local_weights}")
        if missing:
            print(f"[SADRF-R101] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[SADRF-R101] Unexpected keys: {len(unexpected)}")
    return model


class SADRFEnhancedResNet101FeatureExtractor(nn.Module):
    """Non-invasive SADRF sidecar on top of an unchanged ResNet-101 trunk.

    The standard ResNet-101 feature path is kept EXACTLY as in SGRC-Net:
        x1 = Layer1(stem)
        x2 = Layer2(x1)
        x3 = Layer3(x2)
        x4 = Layer4(x3)

    SADRF innovations operate as sidecar enhancements rather than being fed
    back into the backbone stages. This avoids changing the native Stage-3
    graph-node source and Stage-4 semantic anchor before the model has learned
    that the new context is useful.

    Two zero-start adapters then provide controlled interaction:
    - graph node map = x3 + alpha_graph * (enhanced_x3 - x3)
    - CNN vector     = GAP(x4) + alpha_cnn * adaptive_multistage_residual

    Therefore, with alpha_graph=alpha_cnn=0 at initialization, the Chapter-5
    network reproduces the original SGRC-Net feature interfaces exactly while
    retaining a learnable route for SADRF context to help both branches.
    """

    layer3_channels = 1024
    out_channels = 2048

    def __init__(
        self,
        pretrained: bool = True,
        local_weights: str = "",
        fusion_channels: int = 256,
        dropout: float = 0.2,
        stage2_depth: int = 1,
        stage3_depth: int = 2,
        stage2_kernel: int = 5,
        stage3_kernel: int = 9,
        global_residual_init: float = 0.05,
        fusion_local_prior: float = 0.50,
        deploy: bool = False,
        use_graph_context: bool = True,
        graph_context_grad_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(graph_context_grad_scale) <= 1.0:
            raise ValueError("graph_context_grad_scale must be in [0,1]")
        self.use_graph_context = bool(use_graph_context)
        self.graph_context_grad_scale = float(graph_context_grad_scale)

        # This is the same pretrained trunk used by original SGRC-Net.
        model = _build_resnet101(pretrained=pretrained, local_weights=local_weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # Construct every new SADRF module in a forked RNG context. This prevents
        # their random initialization from shifting the initialization sequence
        # of node_proj/FAVOR+/CMRDE/ASRC modules created after the backbone.
        with torch.random.fork_rng(devices=[]):
            self.stage2_context = GlobalDRBContextPath(
                channels=512,
                hidden_channels=128,
                depth=int(stage2_depth),
                large_kernel_size=int(stage2_kernel),
                dilations=(1, 2),
                ffn_expansion=1.0,
                global_residual_init=global_residual_init,
                deploy=deploy,
            )
            self.stage2_fusion = LocalGlobalStageFusion(
                channels=512, local_prior=fusion_local_prior
            )
            self.stage3_context = GlobalDRBContextPath(
                channels=1024,
                hidden_channels=256,
                depth=int(stage3_depth),
                large_kernel_size=int(stage3_kernel),
                dilations=(1, 2, 3),
                ffn_expansion=1.0,
                global_residual_init=global_residual_init,
                deploy=deploy,
            )
            self.stage3_fusion = LocalGlobalStageFusion(
                channels=1024, local_prior=fusion_local_prior
            )
            self.progressive_fusion = ProgressiveStageFusion(
                [256, 512, 1024, 2048], fusion_channels=fusion_channels
            )
            self.semantic_anchor_fusion = SceneAdaptiveResidualFusion(
                original_channels=2048,
                fusion_channels=fusion_channels,
                embed_dim=2048,
                dropout=0.0,
            )

        # Zero-start graph injection is deterministic and consumes no RNG.
        self.graph_context_raw = nn.Parameter(torch.zeros(()))

    def forward_features(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        x = self.stem(x)

        # -------- Pure SGRC-compatible ResNet-101 trunk --------
        x1 = self.layer1(x)
        local_x2 = self.layer2(x1)
        local_x3 = self.layer3(local_x2)
        local_x4 = self.layer4(local_x3)

        # -------- SADRF sidecar context, no feedback into trunk --------
        global_x2 = self.stage2_context(local_x2)
        enhanced_x2, balance2 = self.stage2_fusion(local_x2, global_x2)

        global_x3 = self.stage3_context(local_x3)
        enhanced_x3, balance3 = self.stage3_fusion(local_x3, global_x3)

        # Controlled graph-node enhancement. The forward context can be learned
        # from CNN supervision strongly while GNN gradients into the SADRF
        # sidecar are weakly coupled to reduce objective conflict.
        graph_strength = torch.tanh(self.graph_context_raw)
        if self.use_graph_context:
            graph_delta = enhanced_x3 - local_x3
            delta_detached = graph_delta.detach()
            graph_delta_for_gnn = (
                delta_detached
                + self.graph_context_grad_scale * (graph_delta - delta_detached)
            )
            graph_x3 = local_x3 + graph_strength * graph_delta_for_gnn
        else:
            graph_delta = enhanced_x3 - local_x3
            graph_x3 = local_x3

        # SADRF progressive fusion uses the enhanced Stage-2/Stage-3 sidecar
        # features, but the semantic anchor remains the PURE ResNet Stage-4 map.
        f4, progressive_gates = self.progressive_fusion(
            [x1, enhanced_x2, enhanced_x3, local_x4]
        )
        cnn_embedding, final_gate, cnn_residual_strength = (
            self.semantic_anchor_fusion(local_x4, f4)
        )

        aux: Dict[str, Tensor] = {
            "pure_stage2": local_x2,
            "global_stage2": global_x2,
            "enhanced_stage2": enhanced_x2,
            "stage2_local_global_weights": balance2,
            "pure_stage3": local_x3,
            "global_stage3": global_x3,
            "enhanced_stage3": enhanced_x3,
            "stage3_local_global_weights": balance3,
            "graph_context_delta": graph_delta,
            "graph_context_strength": graph_strength,
            "graph_context_grad_scale": torch.tensor(
                self.graph_context_grad_scale,
                device=local_x3.device,
                dtype=local_x3.dtype,
            ),
            "f4": f4,
            "final_residual_gate": final_gate,
            "cnn_residual_strength": cnn_residual_strength,
            **progressive_gates,
        }
        return graph_x3, local_x4, cnn_embedding, aux

    def forward(self, x: Tensor) -> Tensor:
        # Backward-compatible pure Layer4 feature output.
        _, layer4, _, _ = self.forward_features(x)
        return layer4

    def pretrained_parameters(self):
        """Only original ImageNet ResNet-101 parameters (for low backbone LR)."""
        modules = (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        for module in modules:
            yield from module.parameters()

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        for module in self.modules():
            if isinstance(module, DilatedReparamSpatialBlock):
                module.switch_to_deploy()

