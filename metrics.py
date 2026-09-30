"""Multi-label evaluation metrics used by all experiments.

Reports OP, OR, OF1, CP, CR, CF1 and mAP. The implementation fails fast on
NaN/Inf predictions. Macro metrics are averaged over classes with positive
support in the evaluated split; this is mainly relevant to MLRSNet validation
or any future small diagnostic subset.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _assert_finite_array(name: str, x: np.ndarray) -> None:
    if not np.isfinite(x).all():
        bad = int((~np.isfinite(x)).sum())
        raise FloatingPointError(
            f"{name} contains {bad} NaN/Inf values; metrics are invalid."
        )


def binary_average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    _assert_finite_array("AP scores", scores)

    positive_count = int((targets == 1).sum())
    if positive_count == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]
    tp = np.cumsum(sorted_targets == 1)
    ranks = np.arange(1, len(sorted_targets) + 1, dtype=np.float64)
    precision = tp / ranks
    return float(precision[sorted_targets == 1].sum() / positive_count)


def compute_multilabel_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: probabilities={probabilities.shape}, "
            f"targets={targets.shape}"
        )
    _assert_finite_array("probabilities", probabilities)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("targets must be binary 0/1")

    preds = (probabilities >= threshold).astype(np.int64)
    tp = (preds * targets).sum(axis=0).astype(np.float64)
    fp = (preds * (1 - targets)).sum(axis=0).astype(np.float64)
    fn = ((1 - preds) * targets).sum(axis=0).astype(np.float64)
    support = targets.sum(axis=0).astype(np.int64)
    valid = support > 0

    op = _safe_div(tp.sum(), (tp + fp).sum())
    or_ = _safe_div(tp.sum(), (tp + fn).sum())
    of1 = _safe_div(2 * op * or_, op + or_)

    class_precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) > 0,
    )
    class_recall = np.divide(
        tp,
        tp + fn,
        out=np.full_like(tp, np.nan),
        where=(tp + fn) > 0,
    )
    class_f1 = np.divide(
        2.0 * class_precision * class_recall,
        class_precision + class_recall,
        out=np.zeros_like(class_precision),
        where=np.isfinite(class_recall) & ((class_precision + class_recall) > 0),
    )

    if valid.any():
        cp = float(class_precision[valid].mean())
        cr = float(class_recall[valid].mean())
    else:
        cp = 0.0
        cr = 0.0
    cf1 = _safe_div(2 * cp * cr, cp + cr)

    aps = np.full(targets.shape[1], np.nan, dtype=np.float64)
    for c in np.flatnonzero(valid):
        aps[c] = binary_average_precision(probabilities[:, c], targets[:, c])

    mean_ap = float(np.nanmean(aps)) if valid.any() else 0.0
    scale = 100.0

    return {
        "OP": op * scale,
        "OR": or_ * scale,
        "OF1": of1 * scale,
        "CP": cp * scale,
        "CR": cr * scale,
        "CF1": cf1 * scale,
        "mAP": mean_ap * scale,
        "AP_per_class": [
            None if not np.isfinite(v) else float(v * scale) for v in aps
        ],
        "class_precision": (class_precision * scale).tolist(),
        "class_recall": [
            None if not np.isfinite(v) else float(v * scale)
            for v in class_recall
        ],
        "class_f1": [
            None if not np.isfinite(v) else float(v * scale)
            for v in class_f1
        ],
        "class_support": support.tolist(),
        "valid_macro_classes": int(valid.sum()),
        "num_classes": int(targets.shape[1]),
        "threshold": float(threshold),
    }


class MultiLabelMetricTracker:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._probabilities: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self._names: List[str] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, names=None) -> None:
        logits_d = logits.detach().float()
        if not torch.isfinite(logits_d).all():
            bad = int((~torch.isfinite(logits_d)).sum().item())
            raise FloatingPointError(
                f"logits contain {bad} NaN/Inf values before metric computation"
            )
        probabilities = torch.sigmoid(logits_d).cpu().numpy()
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        self._probabilities.append(probabilities)
        self._targets.append(targets_np)
        if names is not None:
            self._names.extend(list(names))

    def arrays(self):
        if not self._probabilities:
            return (
                np.empty((0, 0)),
                np.empty((0, 0), dtype=np.int64),
            )
        return (
            np.concatenate(self._probabilities, axis=0),
            np.concatenate(self._targets, axis=0),
        )

    def compute(self) -> Dict[str, object]:
        probabilities, targets = self.arrays()
        if probabilities.size == 0:
            return {
                key: 0.0
                for key in ("OP", "OR", "OF1", "CP", "CR", "CF1", "mAP")
            }
        return compute_multilabel_metrics(
            probabilities,
            targets,
            self.threshold,
        )

    @property
    def names(self) -> List[str]:
        return self._names
