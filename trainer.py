"""Training/validation/testing loop for the reset experimental protocol.

Checkpoint policy:
- AID-Multilabel / DFC15-Multilabel: no validation loader exists, therefore
  training always runs to the fixed final epoch and ``final.pth`` is tested.
- MLRSNet: validation is available, therefore ``best.pth`` is selected by
  validation mAP and tested once after training.

The test loader is never used for model selection.
"""
from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from metrics import MultiLabelMetricTracker
from utils import (
    AverageMeter,
    append_csv_row,
    format_metrics,
    load_checkpoint,
    save_checkpoint,
    save_json,
    unwrap_model,
)


METRIC_KEYS = ("mAP", "OP", "OR", "OF1", "CP", "CR", "CF1")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, torch.utils.data.DataLoader],
        class_names: List[str],
        args,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.class_names = list(class_names)
        self.args = args
        self.device = device

        if "train" not in loaders or "test" not in loaders:
            raise ValueError("Trainer requires at least train and test loaders.")

        self.has_validation = "val" in loaders
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model-specific protocol initialization. This is intentionally gated
        # by an explicit model flag, so the original GLR-DRCFNet and every
        # comparison model except ML-GCN follow the exact previous path.
        self._initialize_model_protocol_if_needed()

        self.criterion = nn.BCEWithLogitsLoss()
        base = unwrap_model(model)
        self.optimizer = torch.optim.AdamW(
            base.parameter_groups(args.backbone_lr, args.lr),
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )

        self.amp_enabled = bool(args.amp and device.type == "cuda")
        self.amp_dtype = (
            torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
        )
        scaler_enabled = bool(
            self.amp_enabled and self.amp_dtype == torch.float16
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.start_epoch = 0
        self.best_map = -1.0

        if getattr(args, "detect_anomaly", False):
            torch.autograd.set_detect_anomaly(True)
        if args.resume:
            self._resume(args.resume)

    # ------------------------------------------------------------------
    # Model-specific protocol initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _targets_from_dataset(dataset):
        """Read label targets without loading/augmenting images or advancing RNG.

        The project's RemoteSensingMultiLabelDataset stores deterministic split
        metadata in ``dataset.records``. Reading those records directly avoids
        a second DataLoader pass and therefore does not perturb shuffle order,
        worker seeds, or augmentation random states.
        """
        records = getattr(dataset, "records", None)
        if records is not None:
            rows = [getattr(record, "target", None) for record in records]
            if rows and all(row is not None for row in rows):
                return torch.as_tensor(rows, dtype=torch.float32)

        # Generic compatibility for datasets exposing a targets attribute.
        targets = getattr(dataset, "targets", None)
        if targets is not None:
            tensor = torch.as_tensor(targets, dtype=torch.float32)
            if tensor.ndim == 2:
                return tensor

        # Compatibility with torch.utils.data.Subset-like wrappers while still
        # avoiding __getitem__ and all image transforms.
        base_dataset = getattr(dataset, "dataset", None)
        indices = getattr(dataset, "indices", None)
        if base_dataset is not None and indices is not None:
            base_targets = Trainer._targets_from_dataset(base_dataset)
            if base_targets is not None:
                index_tensor = torch.as_tensor(indices, dtype=torch.long)
                return base_targets.index_select(0, index_tensor)

        return None

    def _initialize_model_protocol_if_needed(self) -> None:
        base = unwrap_model(self.model)
        if not getattr(base, "requires_label_graph_init", False):
            return

        if not hasattr(base, "update_label_graph"):
            raise AttributeError(
                "Model declares requires_label_graph_init=True but does not "
                "implement update_label_graph(targets)."
            )

        targets = self._targets_from_dataset(self.loaders["train"].dataset)
        if targets is None:
            raise RuntimeError(
                "ML-GCN label graph initialization requires direct access to "
                "training targets. The current dataset exposes neither "
                "records nor a 2-D targets attribute."
            )

        if targets.ndim != 2 or targets.shape[1] != len(self.class_names):
            raise ValueError(
                "Training targets used for the label graph must have shape "
                f"[N,{len(self.class_names)}], got {tuple(targets.shape)}"
            )

        base.update_label_graph(targets)
        print(
            f"[Model Init] label graph initialized from "
            f"{targets.shape[0]} training samples."
        )

    # ------------------------------------------------------------------
    # Checkpoint policy
    # ------------------------------------------------------------------

    @property
    def checkpoint_policy(self) -> str:
        return "best_validation_mAP" if self.has_validation else "fixed_final_epoch"

    def default_test_checkpoint(self) -> Path:
        if self.has_validation:
            path = self.output_dir / "best.pth"
        else:
            path = self.output_dir / "final.pth"
        if not path.exists():
            fallback = self.output_dir / "last.pth"
            if fallback.exists():
                return fallback
        return path

    def _resume(self, path: str) -> None:
        checkpoint = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            True,
            "cpu",
        )
        if isinstance(checkpoint, dict):
            self.start_epoch = int(checkpoint.get("epoch", 0))
            self.best_map = float(checkpoint.get("best_map", -1.0))
        print(
            f"[Resume] {path} | start_epoch={self.start_epoch} | "
            f"best_mAP={self.best_map:.2f}"
        )

    # ------------------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_finite_tensor(
        name: str,
        tensor: torch.Tensor,
        split: str,
        epoch: int,
        batch_idx: int,
    ) -> None:
        detached = tensor.detach()
        if torch.isfinite(detached).all():
            return
        t = detached.float()
        finite = t[torch.isfinite(t)]
        stats = (
            "no finite values"
            if finite.numel() == 0
            else f"finite_min={finite.min().item():.4g}, "
                 f"finite_max={finite.max().item():.4g}"
        )
        bad = int((~torch.isfinite(t)).sum().item())
        raise FloatingPointError(
            f"[Numerics] {name} became non-finite at split={split}, "
            f"epoch={epoch}, batch={batch_idx}; bad={bad}; {stats}"
        )

    def _check_output(self, output, split: str, epoch: int, batch_idx: int) -> None:
        important = (
            "logits",
            "cnn_aux_logits",
            "gnn_aux_logits",
            "node_aux_logits",
            "u_rel",
            "u_cls",
            "u_local",
            "u_gain",
            "need_score",
            "gain_score",
            "refinement_score",
            "dynamic_refine_ratio",
            "fused_feature",
        )
        for key in important:
            if key in output and torch.is_tensor(output[key]):
                self._assert_finite_tensor(
                    key, output[key], split, epoch, batch_idx
                )

    # ------------------------------------------------------------------
    # Loss / epoch
    # ------------------------------------------------------------------

    def _compute_loss(self, output, targets):
        y = targets.float()
        main = self.criterion(output["logits"].float(), y)
        aux_losses = []
        for key in ("cnn_aux_logits", "gnn_aux_logits", "node_aux_logits"):
            if key in output:
                aux_losses.append(self.criterion(output[key].float(), y))
        aux_loss = (
            torch.stack(aux_losses).mean()
            if aux_losses
            else torch.zeros_like(main)
        )
        total = main + self.args.aux_weight * aux_loss
        return total, main.detach(), aux_loss.detach()

    def _run_epoch(
        self,
        split: str,
        epoch: int,
        train: bool,
        save_predictions: bool = False,
    ) -> Tuple[float, Dict[str, object], Dict[str, float]]:
        loader = self.loaders[split]
        self.model.train(train)

        meters = {
            key: AverageMeter()
            for key in (
                "loss", "main", "aux", "refine", "qc", "qg", "urel",
                "ucls", "ulocal", "ugain", "need", "gain", "dynratio",
                "selgain", "selneed", "r", "lsgate", "lsent",
            )
        }
        metrics = MultiLabelMetricTracker(self.args.threshold)

        # Lightweight branch diagnostics are collected only for final test
        # evaluation (save_predictions=True), so training speed/memory remain
        # essentially unchanged.  For GLR models:
        #   CNN        -> cnn_logits / cnn_aux_logits
        #   Global/GNN -> gnn_aux_logits
        #   Final      -> logits
        branch_trackers = {}
        if save_predictions and not train:
            branch_trackers["Final"] = metrics
            branch_trackers["CNN"] = MultiLabelMetricTracker(self.args.threshold)
            branch_trackers["Global_GNN"] = MultiLabelMetricTracker(self.args.threshold)

        visualization_buffers = None
        if (
            save_predictions
            and not train
            and self._visualization_enabled()
        ):
            visualization_buffers = self._init_visualization_buffers()

        iterator = tqdm(
            loader,
            desc=f"{'Train' if train else split.capitalize()} {epoch:03d}",
            leave=False,
            disable=self.args.disable_tqdm,
        )

        for batch_idx, batch in enumerate(iterator, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            names = batch["name"]

            self._assert_finite_tensor(
                "images", images, split, epoch, batch_idx
            )
            self._assert_finite_tensor(
                "targets", targets, split, epoch, batch_idx
            )

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                ctx = (
                    torch.autocast("cuda", dtype=self.amp_dtype)
                    if self.amp_enabled
                    else nullcontext()
                )
                with ctx:
                    output = self.model(images)

                self._check_output(output, split, epoch, batch_idx)
                loss, main_loss, aux_loss = self._compute_loss(output, targets)
                self._assert_finite_tensor(
                    "total_loss", loss, split, epoch, batch_idx
                )

                if train:
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                    else:
                        loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=(
                            self.args.grad_clip
                            if self.args.grad_clip > 0
                            else float("inf")
                        ),
                    )
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        raise FloatingPointError(
                            f"[Numerics] gradient norm became non-finite at "
                            f"epoch={epoch}, batch={batch_idx}"
                        )

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            bs = images.size(0)
            meters["loss"].update(loss.item(), bs)
            meters["main"].update(main_loss.item(), bs)
            meters["aux"].update(aux_loss.item(), bs)

            diagnostic_pairs = (
                ("refinement_mask", "refine"),
                ("q_c", "qc"),
                ("q_g", "qg"),
                ("u_rel", "urel"),
                ("u_cls", "ucls"),
                ("u_local", "ulocal"),
                ("u_gain", "ugain"),
                ("need_score", "need"),
                ("gain_score", "gain"),
                ("dynamic_refine_ratio", "dynratio"),
                ("selected_gain_mean", "selgain"),
                ("selected_need_mean", "selneed"),
                ("refinement_score", "r"),
                ("ls_scna_gate", "lsgate"),
                ("ls_scna_assignment_entropy", "lsent"),
            )
            for output_key, meter_key in diagnostic_pairs:
                if output_key in output:
                    meters[meter_key].update(
                        output[output_key].detach().float().mean().item(), bs
                    )

            metrics.update(output["logits"], targets, names)

            if branch_trackers:
                # Prefer the explicit preserved CNN logits when available.
                cnn_logits = output.get("cnn_logits", output.get("cnn_aux_logits"))
                if torch.is_tensor(cnn_logits):
                    branch_trackers["CNN"].update(cnn_logits, targets)

                gnn_logits = output.get("gnn_aux_logits")
                if torch.is_tensor(gnn_logits):
                    branch_trackers["Global_GNN"].update(gnn_logits, targets)

            if visualization_buffers is not None:
                self._collect_visualization_batch(
                    visualization_buffers,
                    output,
                    targets,
                    names,
                )

            if not self.args.disable_tqdm:
                iterator.set_postfix(loss=f"{meters['loss'].avg:.4f}")

        result = metrics.compute()
        diagnostics = {
            "loss": meters["loss"].avg,
            "main_loss": meters["main"].avg,
            "aux_loss": meters["aux"].avg,
            "selected_node_ratio": meters["refine"].avg,
            "mean_q_c": meters["qc"].avg,
            "mean_q_g": meters["qg"].avg,
            "mean_u_rel": meters["urel"].avg,
            "mean_u_cls": meters["ucls"].avg,
            "mean_u_local": meters["ulocal"].avg,
            "mean_u_gain": meters["ugain"].avg,
            "mean_need_score": meters["need"].avg,
            "mean_gain_score": meters["gain"].avg,
            "mean_dynamic_refine_ratio": meters["dynratio"].avg,
            "mean_selected_gain": meters["selgain"].avg,
            "mean_selected_need": meters["selneed"].avg,
            "mean_refinement_score": meters["r"].avg,
            "mean_ls_scna_gate": meters["lsgate"].avg,
            "mean_ls_scna_assignment_entropy": meters["lsent"].avg,
        }
        if branch_trackers:
            branch_map = {"Final": float(result["mAP"])}
            for branch_name in ("CNN", "Global_GNN"):
                tracker = branch_trackers[branch_name]
                probabilities, _ = tracker.arrays()
                if probabilities.size > 0:
                    branch_map[branch_name] = float(tracker.compute()["mAP"])
            diagnostics["branch_mAP"] = branch_map

        if save_predictions:
            self._save_predictions(split, metrics)
            self._save_per_class_metrics(split, result)
            if "branch_mAP" in diagnostics:
                self._save_branch_map(split, diagnostics["branch_mAP"])
            if visualization_buffers is not None:
                self._save_visualization_outputs(
                    split,
                    visualization_buffers,
                )
        return meters["loss"].avg, result, diagnostics


    # ------------------------------------------------------------------
    # Visualization data export (Sec. 3.4.1--3.4.3)
    # ------------------------------------------------------------------

    def _visualization_enabled(self) -> bool:
        """Whether detailed test-time visualization diagnostics should be saved."""
        return bool(getattr(self.args, "export_visualization_data", True))

    def _init_visualization_buffers(self) -> Dict[str, Any]:
        """
        Create CPU-side buffers for the final test pass.

        Two granularities are saved:
        1) label/sample-level outputs for ALL test samples (3.4.1 and 3.4.3);
        2) node-level maps for up to ``visualization_node_limit`` samples
           (3.4.2). 0 means no limit.

        This keeps AID/DFC15 fully exportable while avoiding multi-GB node-map
        dumps on the much larger MLRSNet test split.
        """
        return {
            "seen": 0,
            "names": [],
            "targets": [],
            "label_tensors": {},
            "sample_tensors": {},
            "node_names": [],
            "node_global_indices": [],
            "node_tensors": {},
        }

    @staticmethod
    def _append_tensor(
        store: Dict[str, List[torch.Tensor]],
        key: str,
        tensor: torch.Tensor,
    ) -> None:
        store.setdefault(key, []).append(tensor.detach().cpu())

    def _collect_visualization_batch(
        self,
        buffers: Dict[str, Any],
        output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        names,
    ) -> None:
        """Collect all diagnostics needed by Sec. 3.4 visualizations."""
        bs = int(targets.size(0))
        start_index = int(buffers["seen"])
        buffers["seen"] += bs

        if isinstance(names, (list, tuple)):
            batch_names = [str(x) for x in names]
        else:
            try:
                batch_names = [str(x) for x in list(names)]
            except Exception:
                batch_names = [
                    f"test_{start_index + i:06d}"
                    for i in range(bs)
                ]

        if len(batch_names) != bs:
            batch_names = [
                (
                    batch_names[i]
                    if i < len(batch_names)
                    else f"test_{start_index + i:06d}"
                )
                for i in range(bs)
            ]

        buffers["names"].extend(batch_names)
        buffers["targets"].append(targets.detach().float().cpu())

        # --------------------------------------------------------------
        # 3.4.1: branch/final probabilities for every test sample
        # --------------------------------------------------------------
        final_logits = output.get("logits")
        cnn_logits = output.get("cnn_logits", output.get("cnn_aux_logits"))
        gnn_logits = output.get("gnn_aux_logits")
        adaptive_base_logits = output.get("adaptive_base_logits")

        if torch.is_tensor(final_logits):
            self._append_tensor(
                buffers["label_tensors"],
                "final_prob",
                torch.sigmoid(final_logits.float()),
            )
        if torch.is_tensor(cnn_logits):
            self._append_tensor(
                buffers["label_tensors"],
                "cnn_prob",
                torch.sigmoid(cnn_logits.float()),
            )
        if torch.is_tensor(gnn_logits):
            self._append_tensor(
                buffers["label_tensors"],
                "gnn_prob",
                torch.sigmoid(gnn_logits.float()),
            )
        if torch.is_tensor(adaptive_base_logits):
            self._append_tensor(
                buffers["label_tensors"],
                "adaptive_base_prob",
                torch.sigmoid(adaptive_base_logits.float()),
            )

        if torch.is_tensor(output.get("correction_logits")):
            self._append_tensor(
                buffers["label_tensors"],
                "correction_logits",
                output["correction_logits"].float(),
            )

        # --------------------------------------------------------------
        # 3.4.3: ASRC label-wise internal decision quantities
        # --------------------------------------------------------------
        for key in (
            "label_graph_anchor_weight",
            "label_correction_demand",
            "label_disagreement",
            "label_graph_support",
            "label_cnn_uncertainty",
            "label_gnn_uncertainty",
            "label_demand_learned_gate",
        ):
            value = output.get(key)
            if torch.is_tensor(value):
                self._append_tensor(
                    buffers["label_tensors"],
                    key,
                    value.float(),
                )

        # Small per-sample quantities are kept for all test samples.
        for key in (
            "dynamic_refine_ratio",
            "adaptive_refine_threshold",
            "selected_need_mean",
        ):
            value = output.get(key)
            if torch.is_tensor(value):
                self._append_tensor(
                    buffers["sample_tensors"],
                    key,
                    value.float(),
                )

        # --------------------------------------------------------------
        # 3.4.2: node-level diagnostic maps.
        # --------------------------------------------------------------
        limit = int(getattr(self.args, "visualization_node_limit", 5000))
        already = len(buffers["node_names"])
        remaining = bs if limit <= 0 else max(limit - already, 0)
        take = min(bs, remaining)

        if take <= 0:
            return

        buffers["node_names"].extend(batch_names[:take])
        buffers["node_global_indices"].extend(
            range(start_index, start_index + take)
        )

        for key in (
            "u_rel",
            "u_cls",
            "u_local",
            "refinement_score",
            "refinement_score_cmrde",
            "semantic_relevance",
            "refinement_mask",
            "evidence_weights",
            "selected_indices",
            "selected_active_mask",
        ):
            value = output.get(key)
            if torch.is_tensor(value):
                self._append_tensor(
                    buffers["node_tensors"],
                    key,
                    value[:take],
                )

    @staticmethod
    def _concat_tensor_lists(
        mapping: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for key, parts in mapping.items():
            if not parts:
                continue
            try:
                result[key] = torch.cat(parts, dim=0)
            except Exception as exc:
                print(
                    f"[Visualization][Warning] Could not concatenate {key}: {exc}"
                )
        return result

    def _save_visualization_index_csv(
        self,
        split: str,
        payload: Dict[str, Any],
    ) -> Path:
        """Save a human-readable sample index useful for automatic case selection."""
        path = self.output_dir / f"{split}_visualization_index.csv"

        names = payload["names"]
        targets = payload["targets"].bool()
        label = payload["label_tensors"]
        sample = payload["sample_tensors"]
        threshold = float(self.args.threshold)

        cnn_prob = label.get("cnn_prob")
        gnn_prob = label.get("gnn_prob")
        final_prob = label.get("final_prob")

        def _pred(prob, i):
            return (
                prob[i] >= threshold
                if torch.is_tensor(prob)
                else torch.zeros_like(targets[i], dtype=torch.bool)
            )

        def _label_names(mask: torch.Tensor) -> str:
            ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            return ";".join(
                self.class_names[j]
                for j in ids
                if 0 <= j < len(self.class_names)
            )

        fields = [
            "index",
            "image",
            "ground_truth",
            "cnn_prediction",
            "gnn_prediction",
            "final_prediction",
            "cnn_errors",
            "gnn_errors",
            "final_errors",
            "cnn_recovered_fn",
            "gnn_suppressed_fp",
            "dynamic_refine_ratio",
            "mean_graph_anchor_weight",
            "mean_correction_demand",
        ]

        anchor = label.get("label_graph_anchor_weight")
        demand = label.get("label_correction_demand")
        dyn = sample.get("dynamic_refine_ratio")

        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for i, name in enumerate(names):
                gt = targets[i]
                cnn = _pred(cnn_prob, i)
                gnn = _pred(gnn_prob, i)
                final = _pred(final_prob, i)

                cnn_errors = int((cnn != gt).sum().item())
                gnn_errors = int((gnn != gt).sum().item())
                final_errors = int((final != gt).sum().item())

                cnn_recovered_fn = int(
                    ((~cnn) & gt & final).sum().item()
                )
                gnn_suppressed_fp = int(
                    (gnn & (~gt) & (~final)).sum().item()
                )

                row = {
                    "index": i,
                    "image": str(name),
                    "ground_truth": _label_names(gt),
                    "cnn_prediction": _label_names(cnn),
                    "gnn_prediction": _label_names(gnn),
                    "final_prediction": _label_names(final),
                    "cnn_errors": cnn_errors,
                    "gnn_errors": gnn_errors,
                    "final_errors": final_errors,
                    "cnn_recovered_fn": cnn_recovered_fn,
                    "gnn_suppressed_fp": gnn_suppressed_fp,
                    "dynamic_refine_ratio": (
                        f"{float(dyn[i].item()):.6f}"
                        if torch.is_tensor(dyn) and i < dyn.size(0)
                        else ""
                    ),
                    "mean_graph_anchor_weight": (
                        f"{float(anchor[i].float().mean().item()):.6f}"
                        if torch.is_tensor(anchor) and i < anchor.size(0)
                        else ""
                    ),
                    "mean_correction_demand": (
                        f"{float(demand[i].float().mean().item()):.6f}"
                        if torch.is_tensor(demand) and i < demand.size(0)
                        else ""
                    ),
                }
                writer.writerow(row)

        return path

    def _save_visualization_outputs(
        self,
        split: str,
        buffers: Optional[Dict[str, Any]],
    ) -> None:
        """
        Save compact, reusable test-time tensors for all three qualitative
        analyses without storing the 512x512 input tensors themselves.
        """
        if not buffers:
            return

        targets = (
            torch.cat(buffers["targets"], dim=0)
            if buffers["targets"]
            else torch.empty((0, len(self.class_names)), dtype=torch.float32)
        )
        label_tensors = self._concat_tensor_lists(buffers["label_tensors"])
        sample_tensors = self._concat_tensor_lists(buffers["sample_tensors"])
        node_tensors = self._concat_tensor_lists(buffers["node_tensors"])

        # Infer spatial graph grid when it is square (true for the current
        # square-image ResNet Layer3 graph representation).
        node_grid_shape = None
        ref = node_tensors.get("refinement_score")
        if torch.is_tensor(ref) and ref.ndim >= 2 and ref.size(1) > 0:
            n_nodes = int(ref.size(1))
            side = int(round(n_nodes ** 0.5))
            if side * side == n_nodes:
                node_grid_shape = (side, side)

        common_meta = {
            "dataset": str(getattr(self.args, "dataset", "")),
            "threshold": float(self.args.threshold),
            "class_names": list(self.class_names),
        }

        # 3.4.1 + 3.4.3: all test samples.
        label_payload = {
            **common_meta,
            "names": list(buffers["names"]),
            "targets": targets,
            **label_tensors,
            **sample_tensors,
        }
        label_path = self.output_dir / f"{split}_visualization_labels.pt"
        torch.save(label_payload, label_path)

        # 3.4.2: node maps, optionally capped for very large test sets.
        node_payload = {
            **common_meta,
            "names": list(buffers["node_names"]),
            "global_indices": torch.as_tensor(
                buffers["node_global_indices"],
                dtype=torch.long,
            ),
            "node_grid_shape": node_grid_shape,
            **node_tensors,
        }
        node_path = self.output_dir / f"{split}_visualization_nodes.pt"
        torch.save(node_payload, node_path)

        # Easy-to-inspect sample-selection table.
        index_payload = {
            "names": list(buffers["names"]),
            "targets": targets,
            "label_tensors": label_tensors,
            "sample_tensors": sample_tensors,
        }
        csv_path = self._save_visualization_index_csv(split, index_payload)

        print(f"[Output] Visualization labels saved to {label_path}")
        print(f"[Output] Visualization node maps saved to {node_path}")
        print(f"[Output] Visualization index saved to {csv_path}")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_predictions(self, split: str, tracker) -> None:
        probabilities, targets = tracker.arrays()
        path = self.output_dir / f"{split}_predictions.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            fields = (
                ["image"]
                + [f"prob::{name}" for name in self.class_names]
                + [f"target::{name}" for name in self.class_names]
            )
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i in range(probabilities.shape[0]):
                row = {
                    "image": (
                        tracker.names[i]
                        if i < len(tracker.names)
                        else str(i)
                    )
                }
                for c, name in enumerate(self.class_names):
                    row[f"prob::{name}"] = f"{probabilities[i, c]:.8f}"
                    row[f"target::{name}"] = int(targets[i, c])
                writer.writerow(row)
        print(f"[Output] Predictions saved to {path}")

    def _save_per_class_metrics(self, split: str, metrics: Dict[str, object]) -> None:
        """Save compact class-wise diagnostics for model iteration."""
        aps = metrics.get("AP_per_class", [])
        precision = metrics.get("class_precision", [])
        recall = metrics.get("class_recall", [])
        f1 = metrics.get("class_f1", [])
        support = metrics.get("class_support", [])

        path = self.output_dir / f"{split}_per_class.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["class", "support", "AP", "precision", "recall", "F1"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, class_name in enumerate(self.class_names):
                def value(values, default=""):
                    if i >= len(values) or values[i] is None:
                        return default
                    v = values[i]
                    return f"{float(v):.4f}" if isinstance(v, (int, float)) else v

                writer.writerow({
                    "class": class_name,
                    "support": int(support[i]) if i < len(support) else "",
                    "AP": value(aps),
                    "precision": value(precision),
                    "recall": value(recall),
                    "F1": value(f1),
                })
        print(f"[Output] Per-class metrics saved to {path}")

    def _save_branch_map(self, split: str, branch_map: Dict[str, float]) -> None:
        """Save only the three branch mAP values needed for A->B diagnosis."""
        path = self.output_dir / f"{split}_branch_mAP.csv"
        display_names = {
            "CNN": "CNN",
            "Global_GNN": "Global/GNN",
            "Final": "Final",
        }
        order = ("CNN", "Global_GNN", "Final")
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["branch", "mAP"])
            writer.writeheader()
            for key in order:
                if key in branch_map:
                    writer.writerow({
                        "branch": display_names[key],
                        "mAP": f"{float(branch_map[key]):.4f}",
                    })
        print(f"[Output] Branch mAP saved to {path}")

    # ------------------------------------------------------------------
    # Train / test
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, object]:
        args_dict = vars(self.args).copy()
        args_dict["checkpoint_policy"] = self.checkpoint_policy
        args_dict["has_validation"] = self.has_validation
        save_json(args_dict, self.output_dir / "config.json")
        save_json(
            {"class_names": self.class_names},
            self.output_dir / "classes.json",
        )

        if self.start_epoch == 0 and not self.args.resume:
            history_path = self.output_dir / "history.csv"
            if history_path.exists():
                history_path.unlink()

        last_train_metrics: Dict[str, object] = {}

        for epoch in range(self.start_epoch, self.args.epochs):
            e = epoch + 1
            start = time.time()

            train_loss, train_metrics, _ = self._run_epoch(
                "train", e, True
            )
            last_train_metrics = train_metrics

            val_loss = None
            val_metrics = None
            val_diag = None
            is_best = False

            if self.has_validation:
                val_loss, val_metrics, val_diag = self._run_epoch(
                    "val", e, False
                )
                current_map = float(val_metrics["mAP"])
                if not torch.isfinite(torch.tensor(current_map)):
                    raise FloatingPointError("Validation mAP is non-finite")
                is_best = current_map > self.best_map
                self.best_map = max(self.best_map, current_map)

            self.scheduler.step()

            row = {
                "epoch": e,
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_head": self.optimizer.param_groups[1]["lr"],
                "train_loss": train_loss,
                **{
                    f"train_{key}": train_metrics[key]
                    for key in METRIC_KEYS
                },
                "seconds": time.time() - start,
            }

            if self.has_validation and val_metrics is not None and val_diag is not None:
                row.update(
                    {
                        "val_loss": val_loss,
                        **{
                            f"val_{key}": val_metrics[key]
                            for key in METRIC_KEYS
                        },
                        "val_u_rel": val_diag["mean_u_rel"],
                        "val_u_cls": val_diag["mean_u_cls"],
                        "val_u_local": val_diag["mean_u_local"],
                        "val_u_gain": val_diag["mean_u_gain"],
                        "val_need_score": val_diag["mean_need_score"],
                        "val_gain_score": val_diag["mean_gain_score"],
                        "val_dynamic_refine_ratio": val_diag[
                            "mean_dynamic_refine_ratio"
                        ],
                        "val_selected_gain": val_diag["mean_selected_gain"],
                        "val_selected_need": val_diag["mean_selected_need"],
                        "val_refinement_score": val_diag[
                            "mean_refinement_score"
                        ],
                        "val_ls_scna_gate": val_diag["mean_ls_scna_gate"],
                    }
                )

            append_csv_row(self.output_dir / "history.csv", row)

            # ``last.pth`` is always written for recovery/resume.
            save_checkpoint(
                self.output_dir / "last.pth",
                self.model,
                self.optimizer,
                self.scheduler,
                e,
                self.best_map,
                args_dict,
                self.class_names,
            )

            # Only MLRSNet-like protocols with a validation loader have a
            # validation-selected best checkpoint.
            if self.has_validation and is_best:
                save_checkpoint(
                    self.output_dir / "best.pth",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    e,
                    self.best_map,
                    args_dict,
                    self.class_names,
                )

            # AID/DFC use the pre-declared final epoch as the model-selection
            # rule. This file is never chosen using test performance.
            if not self.has_validation and e == self.args.epochs:
                save_checkpoint(
                    self.output_dir / "final.pth",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    e,
                    -1.0,
                    args_dict,
                    self.class_names,
                )

            if self.has_validation:
                print(
                    f"Epoch {e:03d}/{self.args.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}\n"
                    f"  Train: {format_metrics(train_metrics)}\n"
                    f"  Val:   {format_metrics(val_metrics)} | "
                    f"best_mAP={self.best_map:.2f}"
                )
            else:
                print(
                    f"Epoch {e:03d}/{self.args.epochs} | "
                    f"train_loss={train_loss:.4f}\n"
                    f"  Train: {format_metrics(train_metrics)}\n"
                    "  Selection: fixed final epoch (no validation; test not touched)"
                )

        checkpoint = self.default_test_checkpoint()
        return {
            "checkpoint_policy": self.checkpoint_policy,
            "checkpoint": str(checkpoint),
            "selection_mAP": (
                float(self.best_map) if self.has_validation else float("nan")
            ),
            "final_train_mAP": float(
                last_train_metrics.get("mAP", float("nan"))
            ),
        }

    @torch.no_grad()
    def test(self, checkpoint_path: str = ""):
        path = Path(checkpoint_path) if checkpoint_path else self.default_test_checkpoint()
        if not path.exists():
            raise FileNotFoundError(f"Test checkpoint not found: {path}")

        load_checkpoint(path, self.model, strict=True, map_location="cpu")
        self.model.to(self.device)
        loss, metrics, diagnostics = self._run_epoch(
            "test", 0, False, True
        )

        payload = {
            "checkpoint": str(path),
            "checkpoint_policy": self.checkpoint_policy,
            "loss": loss,
            **{key: metrics[key] for key in METRIC_KEYS},
            "AP_per_class": metrics.get("AP_per_class", []),
            "class_precision": metrics.get("class_precision", []),
            "class_recall": metrics.get("class_recall", []),
            "class_f1": metrics.get("class_f1", []),
            "class_support": metrics.get("class_support", []),
            "threshold": self.args.threshold,
            **diagnostics,
        }
        save_json(payload, self.output_dir / "test_metrics.json")

        print(f"\n[Test] checkpoint={path}")
        print("[Test] " + format_metrics(metrics))
        branch_map = diagnostics.get("branch_mAP", {})
        if branch_map:
            parts = []
            if "CNN" in branch_map:
                parts.append(f"CNN={float(branch_map['CNN']):.2f}")
            if "Global_GNN" in branch_map:
                parts.append(f"Global/GNN={float(branch_map['Global_GNN']):.2f}")
            parts.append(f"Final={float(branch_map['Final']):.2f}")
            print("[Test Branch mAP] " + " | ".join(parts))
        print(f"[Test] loss={loss:.4f}")
        print(
            f"[Test] results saved to "
            f"{self.output_dir / 'test_metrics.json'}"
        )
        return metrics
