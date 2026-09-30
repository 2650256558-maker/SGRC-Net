"""Unified A/B/C/D/E ablation runner for the reset GLR-DRCFNet experiments.

Reset principles
----------------
1. Dataset protocol is fixed before model tuning.
2. AID/DFC15 use official train/test splits with no validation subset.
3. MLRSNet keeps the project-standard 40/10/50 train/val/test protocol.
4. Test data are never used to choose checkpoints.
5. A/B/C/D/E share seed, optimizer, augmentation, epochs and runtime settings.

For the current A->B development stage, the default run is B only:
    A: locked pure ResNet101 baseline (available when needed)
    B: current global-relation model under development
C/D/E remain available through --experiments ABCDE, but are not touched here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Must be set before CUDA initialization for deterministic cuBLAS behavior.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

# Optional complexity profiling tools.
# Install with: pip install thop fvcore
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

import time

from dataset import build_dataloaders, normalize_dataset_name
from models import build_model
from models.resnet101_baseline import ResNet101Baseline
from trainer import Trainer
from utils import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datasets"
SPLIT_ROOT = PROJECT_ROOT / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPECTED_LOCAL_PROJECT_NAME = "AAAAA"


# -----------------------------------------------------------------------------
# A baseline is intentionally isolated in models/resnet101_baseline.py.
# Do not re-implement it in this file and do not import a backbone wrapper from
# glr_drcf_net.py; this prevents future B/C/D/E model edits from changing A.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Ablation definitions
# -----------------------------------------------------------------------------


EXPERIMENTS = {
    "A": {
        "name": "A_ResNet101",
        "description": "Pure ResNet101 baseline",
        "pure_cnn": True,
        "use_cmrde": False,
        "use_local_refinement": False,
        "use_ls_scna": False,
        "use_drcf": False,
    },
    "B": {
        "name": "B_FAVOR",
        "description": "ResNet101 + FAVOR+ global relation",
        "pure_cnn": False,
        "use_cmrde": False,
        "use_local_refinement": False,
        "use_ls_scna": False,
        "use_drcf": False,
    },
    "C": {
        "name": "C_CMRDE_Local",
        "description": "B + CMRDE + sparse local refinement",
        "pure_cnn": False,
        "use_cmrde": True,
        "use_local_refinement": True,
        "use_ls_scna": False,
        "use_drcf": False,
    },
    "D": {
        "name": "D_LSSCNA",
        "description": "C + Label-Semantic-Guided SCNA",
        "pure_cnn": False,
        "use_cmrde": True,
        "use_local_refinement": True,
        "use_ls_scna": True,
        "use_drcf": False,
    },
    "E": {
        "name": "E_Full",
        "description": "D + DRCF full model",
        "pure_cnn": False,
        "use_cmrde": True,
        "use_local_refinement": True,
        "use_ls_scna": True,
        "use_drcf": True,
    },
}


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reset A/B/C/D/E ablation runner for GLR-DRCFNet"
    )

    parser.add_argument(
        "--experiments",
        default="E",
        help="Any combination of A/B/C/D/E. Default: B for the current A->B development stage",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        default="train_test",
    )

    # Dataset / split protocol.
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--dataset",
        default="MLRSNet",
        help="AID-Multilabel | DFC15-Multilabel | MLRSNet",
    )
    parser.add_argument("--split-root", default=str(SPLIT_ROOT))
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)

    # Optimization: kept close to the previous stable runner.
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Model: intentionally unchanged in this reset stage.
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--favor-features", type=int, default=64)
    parser.add_argument(
        "--favor-orthogonal-scaling",
        type=int,
        choices=[0, 1],
        default=0,
    )
    parser.add_argument("--refine-ratio", type=float, default=0.25)
    parser.add_argument("--min-refine-nodes", type=int, default=4)
    parser.add_argument(
        "--local-kernel-size",
        type=int,
        choices=[3, 5],
        default=3,
    )
    parser.add_argument("--ls-scna-temperature", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--gnn-correction-grad-scale",
        type=float,
        default=0.25,
        help=(
            "Gradient scale from the E2/E2.1 relation-correction path into the "
            "upstream GNN branch. 0=no correction-path gradient, "
            "1=full gradient. Default: 0.25."
        ),
    )
    parser.add_argument("--backbone-weights", default="")

    # Visualization diagnostics for Sec. 3.4.
    parser.add_argument(
        "--export-visualization-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During final test, save branch probabilities, ASRC label-wise "
            "diagnostics, and SGLR-GNN node-level maps for qualitative analysis."
        ),
    )
    parser.add_argument(
        "--visualization-node-limit",
        type=int,
        default=5000,
        help=(
            "Maximum number of test samples whose node-level maps are saved. "
            "0 saves all. AID/DFC15 are fully covered by the default 5000; "
            "MLRSNet is capped to avoid very large files."
        ),
    )

    # Complexity analysis
    parser.add_argument(
        "--complexity-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Output Params/FLOPs/MACs/FPS/latency/GPU memory statistics. "
            "Requires thop or fvcore for FLOPs."
        ),
    )
    parser.add_argument("--no-pretrained", action="store_true")

    # Runtime.
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")

    # Compatibility / explicit test checkpoint.
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--run-tag",
        default="B0_diag",
        help=(
            "Short tag appended to output folders so successive B versions do not "
            "overwrite each other. Example: B0_diag, B1_readout, B2_favor128."
        ),
    )
    return parser


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------


def _validate_project_paths() -> None:
    print(f"[Path] main.py:      {Path(__file__).resolve()}")
    print(f"[Path] project root: {PROJECT_ROOT}")
    print(f"[Path] data root:    {DATA_ROOT}")
    print(f"[Path] split root:   {SPLIT_ROOT}")
    print(f"[Path] output root:  {OUTPUT_ROOT}")

    if not DATA_ROOT.exists() or not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"datasets directory not found: {DATA_ROOT}\n"
            "Expected local layout: C:\\AAAAA\\datasets\n"
            "Expected server layout: /home/datasets"
        )

    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists() or not models_dir.is_dir():
        raise FileNotFoundError(
            f"models directory not found: {models_dir}"
        )

    if PROJECT_ROOT.drive and PROJECT_ROOT.name != EXPECTED_LOCAL_PROJECT_NAME:
        print(
            f"[Path][Warning] Windows project folder is '{PROJECT_ROOT.name}', "
            f"not '{EXPECTED_LOCAL_PROJECT_NAME}'."
        )


def _resolve_device(device_arg: str) -> torch.device:
    arg = str(device_arg).strip().lower()
    if arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if arg == "cuda":
        arg = "cuda:0"
    if arg.startswith("cuda") and not torch.cuda.is_available():
        print("[Runtime] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    try:
        return torch.device(arg)
    except Exception as exc:
        raise ValueError(f"Invalid --device value: {device_arg}") from exc


def _auto_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def _version_tuple(value: str | None) -> Tuple[int, int]:
    if not value:
        return (0, 0)
    parts = value.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 0)


def _configure_reproducibility(args) -> None:
    if not bool(args.deterministic):
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _configure_cuda(device: torch.device, args) -> None:
    _configure_reproducibility(args)
    torch.set_float32_matmul_precision(args.matmul_precision)

    if device.type != "cuda":
        return

    torch.cuda.set_device(device)
    if bool(args.deterministic):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    cc = torch.cuda.get_device_capability(idx)
    cuda_build = torch.version.cuda
    print(
        f"[GPU] {props.name} | VRAM={props.total_memory/(1024**3):.1f} GB | "
        f"compute capability={cc[0]}.{cc[1]} | PyTorch={torch.__version__} | "
        f"PyTorch CUDA={cuda_build}"
    )

    if cc[0] >= 12 and _version_tuple(cuda_build) < (12, 8):
        raise RuntimeError(
            "Blackwell GPU detected, but PyTorch CUDA build is older than 12.8."
        )


def _parse_experiments(raw: str) -> List[str]:
    cleaned = str(raw).upper().replace(",", "").replace(" ", "")
    selected: List[str] = []
    for ch in cleaned:
        if ch not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment '{ch}'. Valid experiments: A/B/C/D/E."
            )
        if ch not in selected:
            selected.append(ch)
    if not selected:
        raise ValueError("No experiments selected.")
    return selected


def _experiment_output_dir(args, letter: str) -> Path:
    dataset = normalize_dataset_name(args.dataset)
    name = EXPERIMENTS[letter]["name"]
    tag = str(getattr(args, "run_tag", "")).strip()
    suffix = f"_{tag}" if tag else ""
    return OUTPUT_ROOT / "ablations_reset_v3" / f"{dataset}_{name}_seed{args.seed}{suffix}"


def _build_experiment_model(letter: str, args, num_classes: int) -> nn.Module:
    cfg = EXPERIMENTS[letter]
    if cfg["pure_cnn"]:
        return ResNet101Baseline(
            num_classes=num_classes,
            pretrained_backbone=not args.no_pretrained,
            backbone_weights=args.backbone_weights,
        )

    return build_model(
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        favor_features=args.favor_features,
        favor_orthogonal_scaling=args.favor_orthogonal_scaling,
        refine_ratio=args.refine_ratio,
        min_refine_nodes=args.min_refine_nodes,
        local_kernel_size=args.local_kernel_size,
        ls_scna_temperature=args.ls_scna_temperature,
        dropout=args.dropout,
        pretrained_backbone=not args.no_pretrained,
        backbone_weights=args.backbone_weights,
        use_cmrde=bool(cfg["use_cmrde"]),
        use_local_refinement=bool(cfg["use_local_refinement"]),
        use_drcf=bool(cfg["use_drcf"]),
        use_ls_scna=bool(cfg["use_ls_scna"]),
        gnn_correction_grad_scale=args.gnn_correction_grad_scale,
    )


def _print_ablation_plan(selected: List[str]) -> None:
    print("\n" + "=" * 104)
    print("RESET V2 A/B/C/D/E ABLATION PLAN")
    print("=" * 104)
    print(
        f"{'ID':<3} {'ResNet101':<10} {'FAVOR+':<8} {'CMRDE+Local':<13} "
        f"{'LS-SCNA':<9} {'DRCF':<6} Description"
    )
    print("-" * 104)
    for letter in selected:
        cfg = EXPERIMENTS[letter]
        favor = not cfg["pure_cnn"]
        print(
            f"{letter:<3} {'Y':<10} {('Y' if favor else 'N'):<8} "
            f"{('Y' if cfg['use_cmrde'] and cfg['use_local_refinement'] else 'N'):<13} "
            f"{('Y' if cfg['use_ls_scna'] else 'N'):<9} "
            f"{('Y' if cfg['use_drcf'] else 'N'):<6} {cfg['description']}"
        )
    print("=" * 104 + "\n")


def _print_dataset_policy(dataset: str, loaders: Dict[str, object]) -> None:
    if dataset == "AID-Multilabel":
        policy = "official 2400 train / 600 test; no validation"
    elif dataset == "DFC15-Multilabel":
        policy = "official 2673 train / 669 test; no validation"
    else:
        policy = "40% train / 10% validation / 50% test"
    selection = (
        "best validation mAP"
        if "val" in loaders
        else "fixed final epoch (test never used for selection)"
    )
    print(f"[Protocol] {dataset}: {policy}")
    print(f"[Protocol] checkpoint selection: {selection}")



# -----------------------------------------------------------------------------
# Complexity analysis
# -----------------------------------------------------------------------------

def _profile_model_complexity(model: nn.Module, image_size: int, device: torch.device):
    """
    Profile model complexity using a dummy image.
    Outputs:
      - Parameters
      - FLOPs
      - MACs
      - Single-image latency
      - FPS
      - Peak CUDA memory
    """
    result = {
        "params_M": float("nan"),
        "flops_G": float("nan"),
        "macs_G": float("nan"),
        "latency_ms": float("nan"),
        "fps": float("nan"),
        "peak_memory_GB": float("nan"),
    }

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    result["params_M"] = total_params / 1e6

    dummy = torch.randn(
        1, 3, image_size, image_size, device=device
    )

    # FLOPs / MACs
    try:
        if thop_profile is not None:
            macs, params = thop_profile(
                model,
                inputs=(dummy,),
                verbose=False,
            )
            result["flops_G"] = 2 * macs / 1e9
        elif FlopCountAnalysis is not None:
            flops = FlopCountAnalysis(model, dummy).total()
            result["flops_G"] = flops / 1e9
    except Exception as exc:
        print(f"[Complexity][Warning] FLOPs profiling failed: {exc}")

    # Inference latency
    try:
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            repeat = 50
            for _ in range(repeat):
                _ = model(dummy)

            if device.type == "cuda":
                torch.cuda.synchronize()

            latency = (time.time() - start) / repeat * 1000
            result["latency_ms"] = latency
            result["fps"] = 1000.0 / latency

            if device.type == "cuda":
                result["peak_memory_GB"] = (
                    torch.cuda.max_memory_allocated(device)
                    / (1024 ** 3)
                )

    except Exception as exc:
        print(f"[Complexity][Warning] latency profiling failed: {exc}")

    return result


def _print_complexity(info: Dict[str, object], letter: str):
    print(
        f"[Complexity] {letter}: "
        f"Params={info['params_M']:.2f}M | "
        f"FLOPs={info['flops_G']:.2f}G | "
        f"FPS={info['fps']:.2f}"
    )

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def _save_summary(
    args,
    selected: List[str],
    rows: List[Dict[str, object]],
) -> Path:
    dataset = normalize_dataset_name(args.dataset)
    summary_dir = OUTPUT_ROOT / "ablations_reset_v3"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join(selected)
    csv_path = summary_dir / f"{dataset}_{tag}_summary_seed{args.seed}.csv"
    json_path = summary_dir / f"{dataset}_{tag}_summary_seed{args.seed}.json"

    fields = [
        "experiment",
        "name",
        "description",
        "checkpoint_policy",
        "selection_mAP",
        "final_train_mAP",
        "test_mAP",
        "test_OP",
        "test_OR",
        "test_OF1",
        "test_CP",
        "test_CR",
        "test_CF1",
        "checkpoint",
        "output_dir",
        "params_M",
        "flops_G",
        "fps",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n[Ablation] Summary CSV:  {csv_path}")
    print(f"[Ablation] Summary JSON: {json_path}")
    return csv_path


def _fmt(value: object) -> str:
    try:
        x = float(value)
    except Exception:
        return "-"
    return "-" if math.isnan(x) else f"{x:.2f}"


def _print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 126)
    print("FINAL RESET ABLATION RESULTS")
    print("=" * 126)
    print(
        f"{'ID':<3} {'Select':>8} {'mAP':>8} {'OP':>8} {'OR':>8} "
        f"{'OF1':>8} {'CP':>8} {'CR':>8} {'CF1':>8}  Policy / Name"
    )
    print("-" * 126)
    for row in rows:
        print(
            f"{row['experiment']:<3} "
            f"{_fmt(row.get('selection_mAP')):>8} "
            f"{_fmt(row.get('test_mAP')):>8} "
            f"{_fmt(row.get('test_OP')):>8} "
            f"{_fmt(row.get('test_OR')):>8} "
            f"{_fmt(row.get('test_OF1')):>8} "
            f"{_fmt(row.get('test_CP')):>8} "
            f"{_fmt(row.get('test_CR')):>8} "
            f"{_fmt(row.get('test_CF1')):>8}  "
            f"{row['checkpoint_policy']} / {row['name']}"
        )
    print("=" * 126)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def run(args) -> None:
    _validate_project_paths()

    args.dataset = normalize_dataset_name(args.dataset)
    args.data_root = str(Path(args.data_root).expanduser())
    args.split_root = str(Path(args.split_root).expanduser())
    args.num_workers = _auto_num_workers(args.num_workers)
    selected = _parse_experiments(args.experiments)

    if args.resume and len(selected) > 1:
        raise ValueError(
            "Ablation experiments are independent. Use --resume only with one experiment."
        )
    if args.checkpoint and len(selected) > 1:
        raise ValueError(
            "--checkpoint is only meaningful when testing one experiment."
        )

    seed_everything(args.seed, deterministic=args.deterministic)
    device = _resolve_device(args.device)
    _configure_cuda(device, args)

    print("\n" + "=" * 104)
    print("RESET REPRODUCIBLE RUNTIME")
    print("=" * 104)
    print(f"dataset={args.dataset}")
    print(f"seed={args.seed}")
    print(f"deterministic={args.deterministic}")
    print(f"AMP={args.amp} ({args.amp_dtype})")
    print(f"TF32={args.tf32}")
    print(f"num_workers={args.num_workers}")
    print(f"epochs={args.epochs}")
    print("=" * 104 + "\n")

    _print_ablation_plan(selected)
    rows: List[Dict[str, object]] = []

    for run_idx, letter in enumerate(selected, start=1):
        cfg = EXPERIMENTS[letter]
        print("\n" + "#" * 110)
        print(
            f"# Experiment {letter} ({run_idx}/{len(selected)}): "
            f"{cfg['description']}"
        )
        print("#" * 110)

        # Re-seed every independent ablation so initialization, augmentation,
        # and train-loader shuffling begin from the same random state.
        seed_everything(args.seed, deterministic=args.deterministic)

        loaders, class_names = build_dataloaders(
            data_root=args.data_root,
            dataset_name=args.dataset,
            split_root=args.split_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            overwrite_splits=bool(args.overwrite_splits and run_idx == 1),
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        if run_idx == 1:
            _print_dataset_policy(args.dataset, loaders)

        model = _build_experiment_model(letter, args, len(class_names))
        if letter == "A":
            baseline_id = getattr(model, "baseline_id", type(model).__name__)
            print(f"[Ablation] A locked baseline: {baseline_id}")
            print("[Ablation] A is independent of models/glr_drcf_net.py")
        model.to(device)

        complexity_info = {}
        if args.complexity_analysis:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            complexity_info = _profile_model_complexity(
                model,
                args.image_size,
                device,
            )
            _print_complexity(complexity_info, letter)

        if (
            args.data_parallel
            and device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            model = torch.nn.DataParallel(model)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        exp_args = argparse.Namespace(**vars(args))
        exp_args.output_dir = str(_experiment_output_dir(args, letter))
        exp_args.resume = args.resume if len(selected) == 1 else ""
        exp_args.checkpoint = args.checkpoint if len(selected) == 1 else ""
        exp_args.disable_cmrde = not bool(cfg["use_cmrde"])
        exp_args.disable_local_refinement = not bool(
            cfg["use_local_refinement"]
        )
        exp_args.disable_ls_scna = not bool(cfg["use_ls_scna"])
        exp_args.disable_drcf = not bool(cfg["use_drcf"])

        print(
            f"[Ablation] {letter} | params={total_params/1e6:.2f}M | "
            f"trainable={trainable_params/1e6:.2f}M"
        )
        print(f"[Ablation] Output: {exp_args.output_dir}")
        print(
            f"[Ablation] epochs={exp_args.epochs}, batch={exp_args.batch_size}, "
            f"lr={exp_args.lr}, backbone_lr={exp_args.backbone_lr}, "
            f"threshold={exp_args.threshold}"
        )

        trainer = Trainer(model, loaders, class_names, exp_args, device)
        train_info: Dict[str, object] = {
            "checkpoint_policy": trainer.checkpoint_policy,
            "selection_mAP": float("nan"),
            "final_train_mAP": float("nan"),
            "checkpoint": str(trainer.default_test_checkpoint()),
        }
        test_metrics: Dict[str, object] = {}

        if exp_args.mode in ("train", "train_test"):
            train_info = trainer.train()

        if exp_args.mode in ("test", "train_test"):
            checkpoint = exp_args.checkpoint
            if not checkpoint:
                checkpoint = str(train_info.get("checkpoint", ""))
            test_metrics = trainer.test(checkpoint)

        row = {
            "experiment": letter,
            "name": cfg["name"],
            "description": cfg["description"],
            "checkpoint_policy": train_info.get(
                "checkpoint_policy", trainer.checkpoint_policy
            ),
            "selection_mAP": float(
                train_info.get("selection_mAP", float("nan"))
            ),
            "final_train_mAP": float(
                train_info.get("final_train_mAP", float("nan"))
            ),
            "test_mAP": float(test_metrics.get("mAP", float("nan"))),
            "test_OP": float(test_metrics.get("OP", float("nan"))),
            "test_OR": float(test_metrics.get("OR", float("nan"))),
            "test_OF1": float(test_metrics.get("OF1", float("nan"))),
            "test_CP": float(test_metrics.get("CP", float("nan"))),
            "test_CR": float(test_metrics.get("CR", float("nan"))),
            "test_CF1": float(test_metrics.get("CF1", float("nan"))),
            "checkpoint": str(
                train_info.get("checkpoint", trainer.default_test_checkpoint())
            ),
            "output_dir": exp_args.output_dir,
            "params_M": complexity_info.get("params_M", float("nan")),
            "flops_G": complexity_info.get("flops_G", float("nan")),
            "fps": complexity_info.get("fps", float("nan")),
        }
        rows.append(row)

        del trainer
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        _save_summary(args, selected, rows)
        _print_summary(rows)


if __name__ == "__main__":
    run(build_parser().parse_args())
