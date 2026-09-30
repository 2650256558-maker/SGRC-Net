"""Utility helpers shared by the reset training scripts."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def append_csv_row(path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def format_metrics(metrics: Dict[str, Any]) -> str:
    keys = ("mAP", "OP", "OR", "OF1", "CP", "CR", "CF1")
    return " | ".join(
        f"{key}: {float(metrics.get(key, 0.0)):.2f}" for key in keys
    )


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_map: float = -1.0,
    args_dict=None,
    class_names=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "epoch": int(epoch),
        "best_map": float(best_map),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if args_dict is not None:
        payload["args"] = args_dict
    if class_names is not None:
        payload["class_names"] = class_names
    torch.save(payload, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    strict: bool = True,
    map_location="cpu",
):
    checkpoint = torch.load(path, map_location=map_location)
    state = (
        checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    unwrap_model(model).load_state_dict(state, strict=strict)

    if isinstance(checkpoint, dict):
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
