"""Dataset loading and reproducible official split handling.

Protocols used by this reset version:
- AID-Multilabel: official 2400 train / 600 test. No validation split.
- DFC15-Multilabel: official 2673 train / 669 test. No validation split.
- MLRSNet: deterministic random 40% train / 10% validation / 50% test over the
  full dataset, matching the 40/10/50 protocol used by the official repository.

Important:
- AID/DFC15 official training images are never re-split.
- Test sets are never used for checkpoint selection.
- Split caches are written under ``official_protocol_v2`` so old 90/10
  train/validation caches cannot be reused accidentally.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from scipy.io import loadmat
except Exception:  # pragma: no cover
    loadmat = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

AID_LABELS = [
    "airplane",
    "bare-soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile-home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tanks",
    "trees",
    "water",
]

DFC15_LABELS = [
    "impervious",
    "water",
    "clutter",
    "vegetation",
    "building",
    "tree",
    "boat",
    "car",
]

DATASET_ALIASES = {
    "aid": "AID-Multilabel",
    "aid-multilabel": "AID-Multilabel",
    "aid_multilabel": "AID-Multilabel",
    "aid-ml": "AID-Multilabel",
    "dfc15": "DFC15-Multilabel",
    "dfc15-multilabel": "DFC15-Multilabel",
    "dfc15_multilabel": "DFC15-Multilabel",
    "dfc15-ml": "DFC15-Multilabel",
    "mlrsnet": "MLRSNet",
    "mlrs-net": "MLRSNet",
}


@dataclass
class SampleRecord:
    path: str
    target: List[int]
    group: str = ""


class RandomRotate90:
    """Randomly rotate by 0/90/180/270 degrees without interpolation artifacts."""

    def __call__(self, image: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k == 0:
            return image
        return image.rotate(90 * k, expand=False)


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        supported = sorted(set(DATASET_ALIASES.values()))
        raise ValueError(f"Unknown dataset '{name}'. Supported: {supported}")
    return DATASET_ALIASES[key]


def _normalize_token(value: object) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)
    return s


def _canonical_image_key(value: object) -> str:
    """Canonical image identifier used ONLY for image/label matching.

    Examples:
      Airport_81.jpg -> airport81
      airport_81     -> airport81
      storage-tanks_3 -> storagetanks3

    This deliberately ignores folder names and punctuation, but NEVER falls
    back to positional row order.
    """
    raw = str(value).strip().replace("\\", "/")
    basename = Path(raw).name
    stem = Path(basename).stem
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _find_child_case_insensitive(root: Path, wanted: str) -> Path:
    direct = root / wanted
    if direct.exists():
        return direct
    wanted_l = wanted.lower()
    for child in root.iterdir():
        if child.name.lower() == wanted_l:
            return child
    return direct


def _dataset_root_signature(path: Path, dataset_name: str) -> bool:
    """Return True when *path* looks like the actual dataset directory."""
    if not path.exists() or not path.is_dir():
        return False

    try:
        child_names = {p.name.lower() for p in path.iterdir()}
    except OSError:
        return False

    if dataset_name in {"AID-Multilabel", "DFC15-Multilabel"}:
        has_train = "images_tr" in child_names
        has_test = "images_test" in child_names
        has_csv = "multilabel.csv" in child_names or "multlabel.csv" in child_names
        # CSV is required by the strict label-alignment loader.
        return has_train and has_test and has_csv

    if dataset_name == "MLRSNet":
        return "images" in child_names and "labels" in child_names

    return False


def _resolve_dataset_root(data_root: str, dataset_name: str) -> Path:
    """Resolve datasets using the project's FIXED directory layout.

    Expected project layout:

      <project>/datasets/
        AID-Multilabel/
          images_tr/
          images_test/
          multilabel.csv
          multilabel.mat

        DFC15-Multilabel/
          images_tr/
          images_test/
          multilabel.csv
          multilabel.mat

        MLRSNet/
          Images/
          Labels/
          Categories_names.xlsx

    No sibling search, no recursive guessing, and no wrapper auto-detection are
    performed. A wrong directory therefore fails immediately instead of
    silently selecting another dataset.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    base = Path(data_root).expanduser().resolve()

    if not base.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {base}\n"
            "Expected the project datasets directory, e.g. /home/datasets "
            "or C:\\AAAA\\datasets."
        )
    if not base.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {base}")

    layout = {
        "AID-Multilabel": {
            "folder": "AID-Multilabel",
            "required_dirs": ["images_tr", "images_test"],
            "required_files": ["multilabel.csv"],
            "optional_files": ["multilabel.mat"],
        },
        "DFC15-Multilabel": {
            "folder": "DFC15-Multilabel",
            "required_dirs": ["images_tr", "images_test"],
            "required_files": ["multilabel.csv"],
            "optional_files": ["multilabel.mat"],
        },
        "MLRSNet": {
            "folder": "MLRSNet",
            "required_dirs": ["Images", "Labels"],
            "required_files": [],
            "optional_files": ["Categories_names.xlsx"],
        },
    }[dataset_name]

    # Allow data_root itself only when the caller explicitly points at the
    # requested dataset folder; otherwise use data_root/<fixed-folder-name>.
    if _normalize_token(base.name) == _normalize_token(layout["folder"]):
        root = base
    else:
        root = _find_child_case_insensitive(base, layout["folder"])

    if not root.exists() or not root.is_dir():
        actual = sorted(p.name for p in base.iterdir())
        raise FileNotFoundError(
            f"Dataset directory not found for {dataset_name}.\n"
            f"Expected exactly: {base / layout['folder']}\n"
            f"Actual contents under {base}: {actual}"
        )

    missing_dirs = []
    resolved_dirs = {}
    for name in layout["required_dirs"]:
        p = _find_child_case_insensitive(root, name)
        if not p.exists() or not p.is_dir():
            missing_dirs.append(name)
        else:
            resolved_dirs[name] = p

    missing_files = []
    resolved_files = {}
    for name in layout["required_files"]:
        p = _find_child_case_insensitive(root, name)
        if not p.exists() or not p.is_file():
            missing_files.append(name)
        else:
            resolved_files[name] = p

    if missing_dirs or missing_files:
        actual = sorted(p.name for p in root.iterdir())
        raise FileNotFoundError(
            f"Invalid directory structure for {dataset_name}: {root}\n"
            f"Missing required folders: {missing_dirs or 'none'}\n"
            f"Missing required files: {missing_files or 'none'}\n"
            f"Actual contents: {actual}\n"
            "The loader intentionally does not search another dataset folder "
            "or an extra wrapper directory."
        )

    print(f"[Dataset] Dataset selected: {dataset_name}")
    print(f"[Dataset] Fixed dataset root: {root}")
    for name, path in resolved_dirs.items():
        print(f"[Dataset]   {name}: {path}")
    for name, path in resolved_files.items():
        print(f"[Dataset]   {name}: {path}")

    return root

def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    paths = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.as_posix().lower())


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read comma/tab/semicolon CSV-like files robustly."""
    errors = []
    for sep in (None, ",", "\t", ";"):
        try:
            kwargs = {"engine": "python"}
            if sep is not None:
                kwargs["sep"] = sep
            else:
                kwargs["sep"] = None
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception as exc:  # pragma: no cover - depends on external file
            errors.append(str(exc))
    raise RuntimeError(f"Could not parse label file {path}. Errors: {errors[:2]}")


def _numeric_ratio(series: pd.Series) -> float:
    converted = pd.to_numeric(series, errors="coerce")
    return float(converted.notna().mean())


def _choose_filename_column(df: pd.DataFrame) -> Optional[str]:
    priority = []
    for col in df.columns:
        norm = _normalize_token(col)
        if any(token in norm for token in ("image", "filename", "filepath", "file", "name")):
            priority.append(col)
    for col in priority + list(df.columns):
        if _numeric_ratio(df[col]) < 0.5:
            return col
    return None


def _resolve_label_columns(
    df: pd.DataFrame,
    class_names: Sequence[str],
    filename_col: Optional[str],
) -> List[str]:
    """Align label columns by names when possible, otherwise use numeric columns."""
    available = { _normalize_token(c): c for c in df.columns if c != filename_col }
    aligned = []
    all_found = True
    for class_name in class_names:
        key = _normalize_token(class_name)
        if key in available:
            aligned.append(available[key])
        else:
            all_found = False
            break
    if all_found and len(aligned) == len(class_names):
        return aligned

    numeric_cols = [
        c for c in df.columns
        if c != filename_col and _numeric_ratio(df[c]) >= 0.95
    ]
    # Drop obvious row-index columns.
    numeric_cols = [
        c for c in numeric_cols
        if _normalize_token(c) not in {"index", "id", "no", "number", "unnamed:0"}
    ]
    if len(numeric_cols) < len(class_names):
        raise ValueError(
            f"Label file has only {len(numeric_cols)} usable numeric columns, "
            f"but {len(class_names)} classes are expected. Columns={list(df.columns)}"
        )
    return numeric_cols[: len(class_names)]


def _binary_matrix_from_dataframe(
    df: pd.DataFrame,
    label_cols: Sequence[str],
) -> np.ndarray:
    matrix = df.loc[:, list(label_cols)].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    # Accept 0/1, -1/1, True/False and similar formats.
    return (matrix > 0).astype(np.int64)


def _image_key_variants(value: object) -> List[str]:
    """Return safe filename/ID variants for explicit CSV matching only."""
    key = _canonical_image_key(value)
    variants = [key] if key else []
    if key.isdigit():
        normalized = str(int(key))
        if normalized not in variants:
            variants.append(normalized)
    return variants


def _build_mapping_from_csv(
    csv_path: Path,
    class_names: Sequence[str],
    image_paths: Optional[Sequence[Path]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build a strict filename/ID -> target mapping from CSV.

    Supports:
      - ordinary filename columns (AID);
      - numeric image-ID columns (DFC15);
      - headered or headerless CSV files.

    The mapping is verified against actual image filenames. Positional row
    matching is never used.
    """

    variant_to_real: Dict[str, str] = {}
    image_key_set: Optional[set] = None

    if image_paths is not None:
        image_key_set = set()
        for path in image_paths:
            real_key = _canonical_image_key(Path(path).name)
            if not real_key:
                continue
            image_key_set.add(real_key)

            for variant in _image_key_variants(Path(path).name):
                previous = variant_to_real.get(variant)
                if previous is not None and previous != real_key:
                    raise ValueError(
                        "Ambiguous image-ID normalization: "
                        f"{variant!r} refers to both {previous!r} "
                        f"and {real_key!r}."
                    )
                variant_to_real[variant] = real_key

    candidates = []
    parse_errors = []

    # dtype=str preserves leading zeros in numeric identifiers.
    for header_mode in ("infer", None):
        for sep in (None, ",", "\t", ";"):
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=sep,
                    engine="python",
                    header=header_mode,
                    dtype=str,
                )
            except Exception as exc:
                parse_errors.append(
                    f"header={header_mode}, sep={sep!r}: {exc}"
                )
                continue

            if df.shape[1] <= 1:
                continue

            df = df.loc[
                :,
                [
                    c for c in df.columns
                    if not str(c).startswith("Unnamed:")
                ],
            ]

            if df.shape[1] < len(class_names) + 1:
                continue

            filename_col = None
            matched_rows = -1

            if image_key_set is not None:
                for col in df.columns:
                    score = 0
                    for raw in df[col].astype(str).tolist():
                        if any(
                            variant in variant_to_real
                            for variant in _image_key_variants(raw)
                        ):
                            score += 1

                    if score > matched_rows:
                        matched_rows = score
                        filename_col = col

                # Require near-complete verified filename/ID coverage.
                min_required = max(
                    1,
                    int(round(0.90 * len(image_key_set))),
                )
                if matched_rows < min_required:
                    continue
            else:
                filename_col = _choose_filename_column(df)
                if filename_col is None:
                    if df.shape[1] == len(class_names) + 1:
                        filename_col = df.columns[0]
                    else:
                        continue

            try:
                label_cols = _resolve_label_columns(
                    df,
                    class_names,
                    filename_col,
                )
                matrix = _binary_matrix_from_dataframe(
                    df,
                    label_cols,
                )
            except Exception as exc:
                parse_errors.append(
                    f"header={header_mode}, sep={sep!r}, "
                    f"id_col={filename_col!r}: {exc}"
                )
                continue

            mapping: Dict[str, np.ndarray] = {}
            conflict = False

            for raw_name, target in zip(
                df[filename_col].astype(str).tolist(),
                matrix,
            ):
                real_key = None

                if image_key_set is not None:
                    for variant in _image_key_variants(raw_name):
                        real_key = variant_to_real.get(variant)
                        if real_key is not None:
                            break
                    # Ignore a header row parsed as data or unrelated rows.
                    if real_key is None:
                        continue
                else:
                    real_key = _canonical_image_key(raw_name)
                    if not real_key:
                        continue

                if real_key in mapping:
                    if not np.array_equal(mapping[real_key], target):
                        conflict = True
                        break
                else:
                    mapping[real_key] = target

            if conflict:
                continue

            coverage = (
                sum(1 for key in image_key_set if key in mapping)
                if image_key_set is not None
                else len(mapping)
            )

            expected_rows = (
                len(image_key_set)
                if image_key_set is not None
                else len(df)
            )
            row_distance = abs(len(df) - expected_rows)
            header_preference = 1 if header_mode == "infer" else 0

            candidates.append(
                (
                    coverage,
                    -row_distance,
                    header_preference,
                    header_mode,
                    sep,
                    filename_col,
                    mapping,
                    matrix,
                    len(df),
                )
            )

    if not candidates:
        raise ValueError(
            f"{csv_path} could not be aligned to image filenames/IDs. "
            "Tried headered/headerless parsing and numeric identifier columns. "
            f"Parse errors (first 5): {parse_errors[:5]}"
        )

    candidates.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )

    (
        coverage,
        _,
        _,
        header_mode,
        sep,
        filename_col,
        mapping,
        matrix,
        row_count,
    ) = candidates[0]

    if image_key_set is not None and coverage != len(image_key_set):
        missing = sorted(
            key for key in image_key_set
            if key not in mapping
        )
        raise KeyError(
            "STRICT CSV LABEL ALIGNMENT FAILED: "
            f"matched {coverage}/{len(image_key_set)} images. "
            f"Best parse: header={header_mode}, sep={sep!r}, "
            f"id_col={filename_col!r}, rows={row_count}. "
            f"Missing canonical IDs (first 20): {missing[:20]}"
        )

    print(
        "[Dataset] CSV filename/ID alignment OK: "
        f"matched={coverage}"
        + (
            f"/{len(image_key_set)}"
            if image_key_set is not None
            else ""
        )
        + f", header={header_mode}, sep={sep!r}, "
          f"id_col={filename_col!r}, rows={row_count}"
    )

    return mapping, matrix


def _load_matrix_from_mat(mat_path: Path, num_classes: int, expected_rows: int) -> np.ndarray:
    if loadmat is None:
        raise ImportError("scipy is required to read .mat labels when CSV parsing fails.")
    data = loadmat(mat_path)
    candidates = []
    for key, value in data.items():
        if key.startswith("__") or not isinstance(value, np.ndarray) or value.ndim != 2:
            continue
        if not np.issubdtype(value.dtype, np.number):
            continue
        arr = value
        if arr.shape[1] == num_classes:
            candidates.append(arr)
        elif arr.shape[0] == num_classes:
            candidates.append(arr.T)
    if not candidates:
        raise ValueError(
            f"No 2-D numeric label matrix with {num_classes} classes found in {mat_path}. "
            f"Variables: {[k for k in data.keys() if not k.startswith('__')]}"
        )
    arr = min(candidates, key=lambda x: abs(x.shape[0] - expected_rows))
    if arr.shape[0] < expected_rows:
        raise ValueError(
            f"MAT label matrix has {arr.shape[0]} rows but {expected_rows} images were found."
        )
    return (arr[:expected_rows] > 0).astype(np.int64)


def _lookup_target(mapping: Dict[str, np.ndarray], path: Path) -> Optional[np.ndarray]:
    return mapping.get(_canonical_image_key(path.name))

def _build_aid_or_dfc_records(
    dataset_root: Path,
    class_names: Sequence[str],
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    train_dir = _find_child_case_insensitive(dataset_root, "images_tr")
    test_dir = _find_child_case_insensitive(dataset_root, "images_test")
    train_images = _list_images(train_dir)
    test_images = _list_images(test_dir)
    all_images = train_images + test_images

    # Dataset identity / completeness sanity check. This is deliberately
    # based on the known official split sizes so AID and DFC15 cannot be
    # silently interchanged even though their folder structures are similar.
    if len(class_names) == 17:  # AID-Multilabel
        expected_train, expected_test, dataset_label = 2400, 600, "AID-Multilabel"
    elif len(class_names) == 8:  # DFC15-Multilabel
        expected_train, expected_test, dataset_label = 2673, 669, "DFC15-Multilabel"
    else:
        raise ValueError(
            f"Unexpected class count {len(class_names)} for AID/DFC15 loader."
        )

    if len(train_images) != expected_train or len(test_images) != expected_test:
        raise ValueError(
            f"{dataset_label} image-count check failed at {dataset_root}. "
            f"Expected train/test={expected_train}/{expected_test}, "
            f"but found {len(train_images)}/{len(test_images)}. "
            "Check whether the wrong dataset folder was uploaded or whether "
            "the dataset is incomplete."
        )

    print(
        f"[Dataset] Image-count check OK for {dataset_label}: "
        f"train={len(train_images)}, test={len(test_images)}"
    )

    csv_candidates = ["multilabel.csv", "multlabel.csv"]
    csv_path = next(
        (p for name in csv_candidates
         if (p := _find_child_case_insensitive(dataset_root, name)).exists()),
        None,
    )
    if csv_path is None:
        raise FileNotFoundError(
            f"A readable multilabel.csv is required in {dataset_root}. "
            "MAT-only positional matching is intentionally disabled because "
            "the official train/test folders are not in the same row order "
            "as the combined label matrix."
        )

    mapping, _ = _build_mapping_from_csv(
        csv_path,
        class_names,
        image_paths=all_images,
    )

    records: List[SampleRecord] = []
    missing: List[str] = []
    for path in all_images:
        target = _lookup_target(mapping, path)
        if target is None:
            missing.append(path.relative_to(dataset_root).as_posix())
            continue
        rel = path.relative_to(dataset_root).as_posix()
        records.append(
            SampleRecord(
                rel,
                target.astype(int).tolist(),
                group=path.parent.name,
            )
        )

    if missing:
        examples = "\\n  ".join(missing[:20])
        raise KeyError(
            f"STRICT LABEL ALIGNMENT FAILED: {len(missing)} images could not "
            f"be matched to rows in {csv_path.name}.\\n"
            f"Examples:\\n  {examples}\\n"
            "The loader will NOT fall back to sorted-image/row order."
        )

    if len(records) != len(all_images):
        raise AssertionError("Internal record count mismatch after strict mapping.")

    n_train = len(train_images)
    official_train = records[:n_train]
    official_test = records[n_train:]

    print(
        f"[Dataset] Strict filename-label alignment OK: "
        f"train={len(official_train)}, test={len(official_test)}, "
        f"CSV keys={len(mapping)}"
    )
    return official_train, official_test

def _canonical_mlrsnet_labels(label_files: Sequence[Path]) -> List[str]:
    if not label_files:
        raise FileNotFoundError("No CSV files found in MLRSNet/Labels")
    first = _read_csv_flexible(label_files[0])
    filename_col = _choose_filename_column(first)
    labels = [str(c).strip() for c in first.columns if c != filename_col and not str(c).startswith("Unnamed:")]
    # Keep only columns that appear numeric in the actual rows.
    labels = [c for c in labels if _numeric_ratio(first[c]) >= 0.95]
    if len(labels) >= 60:
        return labels[:60]

    # Fallback: union label names across files, preserving first-seen order.
    seen = set(_normalize_token(x) for x in labels)
    for path in label_files[1:]:
        df = _read_csv_flexible(path)
        fcol = _choose_filename_column(df)
        for c in df.columns:
            if c == fcol or str(c).startswith("Unnamed:") or _numeric_ratio(df[c]) < 0.95:
                continue
            key = _normalize_token(c)
            if key not in seen:
                labels.append(str(c).strip())
                seen.add(key)
    return labels


def _build_mlrsnet_records(dataset_root: Path) -> Tuple[List[SampleRecord], List[str]]:
    images_root = _find_child_case_insensitive(dataset_root, "Images")
    labels_root = _find_child_case_insensitive(dataset_root, "Labels")
    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(
            f"MLRSNet expects Images/ and Labels/ under {dataset_root}."
        )

    label_files = sorted(labels_root.glob("*.csv"), key=lambda p: p.name.lower())
    class_names = _canonical_mlrsnet_labels(label_files)
    if len(class_names) != 60:
        print(f"[Dataset] Warning: expected 60 MLRSNet labels, detected {len(class_names)}")

    # Index by basename only as a fallback. The normal path is Images/<category>/<image>.
    image_index: Optional[Dict[str, Path]] = None
    records: List[SampleRecord] = []

    canonical_norm = [_normalize_token(c) for c in class_names]

    for label_file in label_files:
        group = label_file.stem
        df = _read_csv_flexible(label_file)
        filename_col = _choose_filename_column(df)
        if filename_col is None:
            raise ValueError(f"No image-name column found in MLRSNet label file: {label_file}")

        column_map = {_normalize_token(c): c for c in df.columns if c != filename_col}
        missing_cols = [name for name, norm in zip(class_names, canonical_norm) if norm not in column_map]
        if missing_cols:
            raise ValueError(
                f"MLRSNet label file {label_file.name} is missing canonical columns: {missing_cols[:5]}"
            )
        aligned_cols = [column_map[norm] for norm in canonical_norm]
        matrix = _binary_matrix_from_dataframe(df, aligned_cols)

        group_dir = _find_child_case_insensitive(images_root, group)
        for raw_name, target in zip(df[filename_col].astype(str).tolist(), matrix):
            basename = Path(raw_name.strip().replace("\\", "/")).name
            path = group_dir / basename
            if not path.exists():
                if image_index is None:
                    image_index = {p.name.lower(): p for p in _list_images(images_root)}
                path = image_index.get(basename.lower(), path)
            if not path.exists():
                raise FileNotFoundError(
                    f"MLRSNet image listed in {label_file.name} was not found: {basename}"
                )
            rel = path.relative_to(dataset_root).as_posix()
            records.append(SampleRecord(rel, target.astype(int).tolist(), group=group))

    return records, class_names


def _split_train_val(
    records: Sequence[SampleRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    n_val = int(round(len(records) * val_ratio))
    n_val = min(max(n_val, 1), max(len(records) - 1, 1))
    val_idx = set(indices[:n_val].tolist())
    train = [record for i, record in enumerate(records) if i not in val_idx]
    val = [record for i, record in enumerate(records) if i in val_idx]
    return train, val



def _split_train_val_by_group(
    records: Sequence[SampleRecord],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    """Stratify AID train/val split by scene-category folder."""
    groups: Dict[str, List[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.group, []).append(record)

    train: List[SampleRecord] = []
    val: List[SampleRecord] = []
    for group_name in sorted(groups):
        items = list(groups[group_name])
        rng = np.random.RandomState(seed + sum(ord(c) for c in group_name))
        order = np.arange(len(items))
        rng.shuffle(order)
        n_val = int(round(len(items) * val_ratio))
        n_val = min(max(n_val, 1), max(len(items) - 1, 1))
        val_ids = set(order[:n_val].tolist())
        train.extend([r for i, r in enumerate(items) if i not in val_ids])
        val.extend([r for i, r in enumerate(items) if i in val_ids])

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val

def _split_mlrsnet_official_ratio(
    records: Sequence[SampleRecord],
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    """Deterministic 40/10/50 random split over the full MLRSNet dataset.

    The official repository states that its released trained models use a
    40%/10%/50% training/validation/testing ratio. The original paper describes
    random selection under this ratio. We therefore shuffle the complete sample
    list once with the experiment seed and slice it into 40/10/50 subsets.
    """
    items = list(records)
    if len(items) < 3:
        raise ValueError("MLRSNet split requires at least three samples.")

    rng = np.random.RandomState(seed)
    order = np.arange(len(items))
    rng.shuffle(order)
    shuffled = [items[i] for i in order]

    n = len(shuffled)
    n_train = int(round(n * 0.40))
    n_val = int(round(n * 0.10))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def _write_split(path: Path, records: Sequence[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({"path": record.path, "target": record.target, "group": record.group}, ensure_ascii=False) + "\n")


def _read_split(path: Path) -> List[SampleRecord]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(SampleRecord(item["path"], list(map(int, item["target"])), item.get("group", "")))
    return records



def _validate_existing_aid_dfc_splits(
    dataset_root: Path,
    dataset_name: str,
    splits: Dict[str, List[SampleRecord]],
) -> Tuple[bool, int]:
    """Validate cached split targets against the current CSV by filename."""
    class_names = AID_LABELS if dataset_name == "AID-Multilabel" else DFC15_LABELS
    official_train, official_test = _build_aid_or_dfc_records(dataset_root, class_names)
    source = {r.path: r.target for r in official_train + official_test}

    mismatches = 0
    for split_name, records in splits.items():
        for record in records:
            expected = source.get(record.path)
            if expected is None or list(map(int, record.target)) != list(map(int, expected)):
                mismatches += 1
                if mismatches <= 5:
                    print(
                        f"[Dataset] Cached split mismatch [{split_name}]: "
                        f"{record.path}"
                    )
    return mismatches == 0, mismatches

def _validate_official_aid_dfc_cache(
    dataset_root: Path,
    dataset_name: str,
    splits: Dict[str, List[SampleRecord]],
) -> bool:
    """Validate that cached AID/DFC15 splits exactly match official folders."""
    class_names = AID_LABELS if dataset_name == "AID-Multilabel" else DFC15_LABELS
    official_train, official_test = _build_aid_or_dfc_records(dataset_root, class_names)

    if set(splits) != {"train", "test"}:
        return False

    train_expected = {r.path: r.target for r in official_train}
    test_expected = {r.path: r.target for r in official_test}
    train_cached = {r.path: r.target for r in splits["train"]}
    test_cached = {r.path: r.target for r in splits["test"]}

    return train_cached == train_expected and test_cached == test_expected


def prepare_splits(
    data_root: str,
    dataset_name: str,
    split_root: str = "./splits",
    seed: int = 42,
    overwrite: bool = False,
) -> Tuple[Path, List[str], Dict[str, List[SampleRecord]]]:
    dataset_name = normalize_dataset_name(dataset_name)
    dataset_root = _resolve_dataset_root(data_root, dataset_name)

    # New protocol namespace prevents old AID/DFC 90/10 validation caches from
    # being loaded after this reset.
    split_dir = (
        Path(split_root).expanduser().resolve()
        / "official_protocol_v2"
        / dataset_name
        / f"seed_{seed}"
    )
    train_file = split_dir / "train.jsonl"
    val_file = split_dir / "val.jsonl"
    test_file = split_dir / "test.jsonl"
    classes_file = split_dir / "classes.json"
    protocol_file = split_dir / "protocol.json"

    needs_val = dataset_name == "MLRSNet"
    required = [train_file, test_file, classes_file, protocol_file]
    if needs_val:
        required.append(val_file)

    if all(p.exists() for p in required) and not overwrite:
        class_names = json.loads(classes_file.read_text(encoding="utf-8"))
        splits: Dict[str, List[SampleRecord]] = {
            "train": _read_split(train_file),
            "test": _read_split(test_file),
        }
        if needs_val:
            splits["val"] = _read_split(val_file)

        if dataset_name in {"AID-Multilabel", "DFC15-Multilabel"}:
            if _validate_official_aid_dfc_cache(dataset_root, dataset_name, splits):
                print("[Dataset] Official train/test cache verified.")
                return dataset_root, class_names, splits
            print("[Dataset] Cached AID/DFC split does not match official folders; rebuilding.")
        else:
            return dataset_root, class_names, splits

    print(f"[Dataset] Building official-protocol splits for {dataset_name} ...")

    if dataset_name == "AID-Multilabel":
        train, test = _build_aid_or_dfc_records(dataset_root, AID_LABELS)
        class_names = AID_LABELS
        splits = {"train": train, "test": test}
        protocol = {
            "dataset": dataset_name,
            "policy": "official_train_test_no_validation",
            "train": 2400,
            "validation": 0,
            "test": 600,
            "seed_affects_split": False,
        }
    elif dataset_name == "DFC15-Multilabel":
        train, test = _build_aid_or_dfc_records(dataset_root, DFC15_LABELS)
        class_names = DFC15_LABELS
        splits = {"train": train, "test": test}
        protocol = {
            "dataset": dataset_name,
            "policy": "official_train_test_no_validation",
            "train": 2673,
            "validation": 0,
            "test": 669,
            "seed_affects_split": False,
        }
    elif dataset_name == "MLRSNet":
        all_records, class_names = _build_mlrsnet_records(dataset_root)
        train, val, test = _split_mlrsnet_official_ratio(all_records, seed=seed)
        splits = {"train": train, "val": val, "test": test}
        protocol = {
            "dataset": dataset_name,
            "policy": "official_repo_random_40_10_50",
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "seed_affects_split": True,
        }
    else:  # pragma: no cover
        raise AssertionError(dataset_name)

    split_dir.mkdir(parents=True, exist_ok=True)
    _write_split(train_file, splits["train"])
    _write_split(test_file, splits["test"])
    if "val" in splits:
        _write_split(val_file, splits["val"])
    elif val_file.exists():
        val_file.unlink()

    classes_file.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    protocol_file.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    print(f"[Dataset] {dataset_name}: {counts}, classes={len(class_names)}")

    if dataset_name == "AID-Multilabel" and (
        len(splits["train"]), len(splits["test"])
    ) != (2400, 600):
        raise AssertionError("AID official split must be exactly 2400/600.")
    if dataset_name == "DFC15-Multilabel" and (
        len(splits["train"]), len(splits["test"])
    ) != (2673, 669):
        raise AssertionError("DFC15 official split must be exactly 2673/669.")

    return dataset_root, class_names, splits


class RemoteSensingMultiLabelDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        records: Sequence[SampleRecord],
        transform=None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.records = list(records)
        self.transform = transform
        if not self.records:
            raise RuntimeError("Dataset split contains no samples.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.dataset_root / Path(record.path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        target = torch.tensor(record.target, dtype=torch.float32)
        return {
            "image": image,
            "target": target,
            "name": record.path,
        }


def build_transforms(image_size: int = 224, is_train: bool = True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                RandomRotate90(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(
    data_root: str,
    dataset_name: str,
    split_root: str = "./splits",
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    overwrite_splits: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[Dict[str, DataLoader], List[str]]:
    dataset_root, class_names, splits = prepare_splits(
        data_root=data_root,
        dataset_name=dataset_name,
        split_root=split_root,
        seed=seed,
        overwrite=overwrite_splits,
    )

    train_ds = RemoteSensingMultiLabelDataset(
        dataset_root, splits["train"], build_transforms(image_size, True)
    )
    test_ds = RemoteSensingMultiLabelDataset(
        dataset_root, splits["test"], build_transforms(image_size, False)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        common["prefetch_factor"] = max(int(prefetch_factor), 1)

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            shuffle=True,
            drop_last=False,
            generator=generator,
            **common,
        ),
        "test": DataLoader(test_ds, shuffle=False, drop_last=False, **common),
    }

    if "val" in splits:
        val_ds = RemoteSensingMultiLabelDataset(
            dataset_root, splits["val"], build_transforms(image_size, False)
        )
        loaders["val"] = DataLoader(
            val_ds, shuffle=False, drop_last=False, **common
        )

    print(
        f"[Dataset] loaders={list(loaders.keys())}; "
        f"checkpoint selection={'validation mAP' if 'val' in loaders else 'fixed final epoch'}"
    )
    return loaders, class_names

