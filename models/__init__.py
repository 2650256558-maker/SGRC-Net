
"""
Model entry points.

This unified registry supports both:
1. Original experiments:
   - ResNet101 baseline
   - GLR-DRCFNet (loaded lazily)

2. Comparison experiments:
   - ML-GCN
   - ADD-GCN
   - MLRSSC-CNN-GNN
   - SIGNA
   - LD-GCN
   - S-MAT
   - CAGRN
   - MSDR-Net
"""

from __future__ import annotations


# =========================
# Original experiment models
# =========================

from .resnet101_baseline import (
    ResNet101Baseline,
    build_resnet101_baseline,
)


def build_model(*args, **kwargs):
    """
    Original GLR-DRCFNet entry point.
    Keep lazy import to avoid unnecessary dependency when running
    comparison experiments.
    """
    from .glr_drcf_net import build_model as _build_model
    return _build_model(*args, **kwargs)


def __getattr__(name):
    if name in {"GLRDRCFConfig", "GLRDRCFNet"}:
        from .glr_drcf_net import (
            GLRDRCFConfig,
            GLRDRCFNet,
        )
        return {
            "GLRDRCFConfig": GLRDRCFConfig,
            "GLRDRCFNet": GLRDRCFNet,
        }[name]

    raise AttributeError(name)


# =========================
# Comparison experiment models
# =========================

from .ml_gcn import build_ml_gcn
from .add_gcn import build_add_gcn
from .mlrssc_cnn_gnn import build_mlrssc_cnn_gnn
from .signa import build_signa
from .ld_gcn import build_ld_gcn
from .smat import build_smat
from .cagrn import build_cagrn
from .msdr_net import build_msdr_net


COMPARISON_MODEL_FACTORY = {

    "resnet101":
        build_resnet101_baseline,

    "ml_gcn":
        build_ml_gcn,

    "add_gcn":
        build_add_gcn,

    "mlrssc_cnn_gnn":
        build_mlrssc_cnn_gnn,

    "signa":
        build_signa,

    "ld_gcn":
        build_ld_gcn,

    "smat":
        build_smat,

    "cagrn":
        build_cagrn,

    "msdr_net":
        build_msdr_net,
}


def build_comparison_model(
    name,
    num_classes,
    **kwargs
):
    """
    Build comparison model by name.
    """

    if name not in COMPARISON_MODEL_FACTORY:
        raise ValueError(
            f"Unknown comparison model: {name}. "
            f"Available: "
            f"{list(COMPARISON_MODEL_FACTORY.keys())}"
        )

    return COMPARISON_MODEL_FACTORY[name](
        num_classes=num_classes,
        **kwargs
    )


__all__ = [

    # original
    "ResNet101Baseline",
    "build_resnet101_baseline",
    "build_model",
    "GLRDRCFConfig",
    "GLRDRCFNet",

    # comparison
    "build_comparison_model",
    "COMPARISON_MODEL_FACTORY",

]
