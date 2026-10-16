"""Create and verify a frozen Stratified Group K-fold manifest.

Only rows marked ``keep`` in the deduplication audit are eligible. Existing
``visual_group`` identifiers are indivisible. A retained image without a
visual group receives a deterministic singleton group based on its SHA-256.

Fold numbers in the output are one-based (1..K).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_AUDIT_COLUMNS = {
    "path",
    "class",
    "sha256",
    "status",
    "visual_group",
}


@dataclass(frozen=True)
class FoldRecord:
    path: str
    class_name: str
    label: int
    group_id: str
    fold: int
    sha256: str
    visual_group: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_image_path(data_root: Path, relative_path: str) -> Path:
    root = data_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {relative_path}")
    return candidate


def load_kept_records(
    audit_manifest: Path,
    data_root: Path | None = None,
    verify_files: bool = True,
) -> list[dict[str, str]]:
    """Load, normalize, and validate retained audit rows."""

    audit_manifest = Path(audit_manifest)
    if not audit_manifest.is_file():
        raise FileNotFoundError(f"Audit manifest not found: {audit_manifest}")

    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    with audit_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_AUDIT_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Audit manifest is missing columns: {sorted(missing_columns)}"
            )

        for source in reader:
            if source["status"].strip() != "keep":
                continue
            relative_path = source["path"].strip()
            class_name = source["class"].strip()
            file_sha = source["sha256"].strip().lower()
            visual_group = source["visual_group"].strip()
            if not relative_path or not class_name or not file_sha:
                raise ValueError(
                    "Every kept row must have path, class, and sha256: "
                    f"{source}"
                )
            if relative_path in seen_paths:
                raise ValueError(f"Duplicate kept path: {relative_path}")
            seen_paths.add(relative_path)

            if data_root is not None and verify_files:
                image_path = _safe_image_path(Path(data_root), relative_path)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Materialized retained image not found: {image_path}"
                    )

            rows.append(
                {
                    "path": relative_path,
                    "class_name": class_name,
                    "sha256": file_sha,
                    "visual_group": visual_group,
                    "group_id": visual_group or f"S:{file_sha}",
                }
            )

    if not rows:
        raise ValueError(f"No rows with status=keep in {audit_manifest}")
    rows.sort(key=lambda row: row["path"])
    return rows


def assign_stratified_group_folds(
    rows: Sequence[dict[str, str]],
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[list[FoldRecord], list[str]]:
    """Assign every row to exactly one group-constrained stratified fold."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    rows = sorted(rows, key=lambda row: row["path"])
    classes = sorted({row["class_name"] for row in rows})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    labels = np.asarray([class_to_idx[row["class_name"]] for row in rows])
    groups = np.asarray([row["group_id"] for row in rows], dtype=object)

    groups_per_class: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_per_class[row["class_name"]].add(row["group_id"])
    insufficient = {
        name: len(group_ids)
        for name, group_ids in groups_per_class.items()
        if len(group_ids) < n_splits
    }
    if insufficient:
        raise ValueError(
            "Each class must occur in at least n_splits distinct groups: "
            f"{insufficient}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    fold_by_index = np.zeros(len(rows), dtype=np.int16)
    dummy_x = np.zeros((len(rows), 1), dtype=np.uint8)
    for fold_number, (_, validation_indices) in enumerate(
        splitter.split(dummy_x, labels, groups),
        start=1,
    ):
        fold_by_index[validation_indices] = fold_number

    if np.any(fold_by_index == 0):
        missing = int(np.sum(fold_by_index == 0))
        raise AssertionError(f"{missing} rows were not assigned to a fold")

    records = [
        FoldRecord(
            path=row["path"],
            class_name=row["class_name"],
            label=class_to_idx[row["class_name"]],
            group_id=row["group_id"],
            fold=int(fold_by_index[index]),
            sha256=row["sha256"],
            visual_group=row["visual_group"],
        )
        for index, row in enumerate(rows)
    ]
    validate_fold_records(records, classes, n_splits)
    return records, classes


def validate_fold_records(
    records: Sequence[FoldRecord],
    classes: Sequence[str],
    n_splits: int,
) -> None:
    """Fail if a row/group leaks or a class is absent from any fold."""

    expected_folds = set(range(1, n_splits + 1))
    folds = {record.fold for record in records}
    if folds != expected_folds:
        raise AssertionError(f"Expected folds {expected_folds}, found {folds}")

    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise AssertionError("A retained path occurs more than once")

    group_folds: dict[str, set[int]] = defaultdict(set)
    fold_classes: dict[int, set[str]] = defaultdict(set)
    for record in records:
        group_folds[record.group_id].add(record.fold)
        fold_classes[record.fold].add(record.class_name)
    leaking = [group for group, assigned in group_folds.items() if len(assigned) != 1]
    if leaking:
        raise AssertionError(
            f"{len(leaking)} group(s) cross folds; first={leaking[:5]}"
        )

    expected_classes = set(classes)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    label_errors = [
        record.path
        for record in records
        if class_to_idx.get(record.class_name) != record.label
    ]
    if label_errors:
        raise AssertionError(
            f"{len(label_errors)} row(s) have an invalid class index; "
            f"first={label_errors[:5]}"
        )
    for fold in expected_folds:
        missing = expected_classes - fold_classes[fold]
        if missing:
            raise AssertionError(
                f"Fold {fold} is missing classes: {sorted(missing)}"
            )


def build_summary(
    records: Sequence[FoldRecord],
    classes: Sequence[str],
    n_splits: int,
    seed: int,
    audit_manifest: Path,
    data_root: Path,
) -> dict:
    per_fold: dict[str, dict] = {}
    for fold in range(1, n_splits + 1):
        subset = [record for record in records if record.fold == fold]
        counts = Counter(record.class_name for record in subset)
        per_fold[str(fold)] = {
            "validation_images": len(subset),
            "training_images": len(records) - len(subset),
            "validation_groups": len({record.group_id for record in subset}),
            "class_counts": {name: counts[name] for name in classes},
        }

    return {
        "schema_version": 1,
        "method": "sklearn.model_selection.StratifiedGroupKFold",
        "scikit_learn_version": sklearn.__version__,
        "fold_numbering": "one-based",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": seed,
        "audit_manifest": str(Path(audit_manifest).resolve()),
        "audit_manifest_sha256": sha256_file(Path(audit_manifest)),
        "data_root": str(Path(data_root).resolve()),
        "images": len(records),
        "groups": len({record.group_id for record in records}),
        "visual_group_images": sum(bool(record.visual_group) for record in records),
        "singleton_images": sum(not record.visual_group for record in records),
        "classes": list(classes),
        "class_to_idx": {name: idx for idx, name in enumerate(classes)},
        "per_fold": per_fold,
        "invariants": {
            "each_image_exactly_one_fold": True,
            "each_group_exactly_one_fold": True,
            "all_classes_in_every_fold": True,
        },
    }


def _atomic_write_csv(path: Path, records: Iterable[FoldRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [
        "path",
        "class",
        "label",
        "group_id",
        "fold",
        "sha256",
        "visual_group",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": record.path,
                    "class": record.class_name,
                    "label": record.label,
                    "group_id": record.group_id,
                    "fold": record.fold,
                    "sha256": record.sha256,
                    "visual_group": record.visual_group,
                }
            )
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def create_fold_manifest(
    audit_manifest: Path,
    data_root: Path,
    output_manifest: Path,
    output_summary: Path,
    n_splits: int = 5,
    seed: int = 42,
    force: bool = False,
    verify_files: bool = True,
) -> tuple[list[FoldRecord], dict]:
    """Create a fold manifest, or reuse it only when provenance matches."""

    audit_manifest = Path(audit_manifest)
    data_root = Path(data_root)
    output_manifest = Path(output_manifest)
    output_summary = Path(output_summary)

    if output_manifest.exists() or output_summary.exists():
        if not (output_manifest.exists() and output_summary.exists()):
            raise FileExistsError(
                "Fold manifest and summary must either both exist or both be absent"
            )
        if not force:
            records, summary = load_fold_manifest(output_manifest, output_summary)
            expected_hash = sha256_file(audit_manifest)
            matching = (
                summary.get("n_splits") == n_splits
                and summary.get("random_state") == seed
                and summary.get("audit_manifest_sha256") == expected_hash
                and Path(summary.get("data_root", "")).resolve()
                == data_root.resolve()
            )
            if not matching:
                raise FileExistsError(
                    "Existing frozen folds have different provenance. Use "
                    "--force only when you intentionally want a new assignment."
                )
            return records, summary

    rows = load_kept_records(
        audit_manifest,
        data_root=data_root,
        verify_files=verify_files,
    )
    records, classes = assign_stratified_group_folds(rows, n_splits, seed)
    summary = build_summary(
        records,
        classes,
        n_splits,
        seed,
        audit_manifest,
        data_root,
    )
    _atomic_write_csv(output_manifest, records)
    summary["fold_manifest"] = str(output_manifest.resolve())
    summary["fold_manifest_sha256"] = sha256_file(output_manifest)
    _atomic_write_json(output_summary, summary)
    return records, summary


def load_fold_manifest(
    manifest_path: Path,
    summary_path: Path | None = None,
) -> tuple[list[FoldRecord], dict]:
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = [
            FoldRecord(
                path=row["path"],
                class_name=row["class"],
                label=int(row["label"]),
                group_id=row["group_id"],
                fold=int(row["fold"]),
                sha256=row["sha256"],
                visual_group=row["visual_group"],
            )
            for row in reader
        ]
    if not records:
        raise ValueError(f"Empty fold manifest: {manifest_path}")

    if summary_path is not None:
        with Path(summary_path).open(encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        classes = sorted({record.class_name for record in records})
        summary = {
            "n_splits": max(record.fold for record in records),
            "classes": classes,
        }

    classes = list(summary["classes"])
    validate_fold_records(records, classes, int(summary["n_splits"]))
    if summary_path is not None:
        recorded_hash = summary.get("fold_manifest_sha256")
        current_hash = sha256_file(manifest_path)
        if recorded_hash and recorded_hash != current_hash:
            raise ValueError(
                f"Fold manifest hash mismatch: {recorded_hash} != {current_hash}"
            )
        if int(summary.get("images", len(records))) != len(records):
            raise ValueError("Fold manifest row count differs from its summary")
    return records, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=project_root / "dedup_audit_v3_verified" / "dedup_manifest.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data_deduplicated_v3",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root / "cv_splits" / "stratified_group_5fold.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=project_root / "cv_splits" / "stratified_group_5fold_summary.json",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Do not verify that every retained image exists under data-root.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    records, summary = create_fold_manifest(
        audit_manifest=args.audit_manifest,
        data_root=args.data_root,
        output_manifest=args.output_manifest,
        output_summary=args.output_summary,
        n_splits=args.n_splits,
        seed=args.seed,
        force=args.force,
        verify_files=not args.skip_file_check,
    )
    print(
        f"[folds] images={len(records):,} groups={summary['groups']:,} "
        f"folds={summary['n_splits']} seed={summary['random_state']}"
    )
    for fold, details in summary["per_fold"].items():
        print(
            f"  fold {fold}: val={details['validation_images']:,} "
            f"train={details['training_images']:,} "
            f"val_groups={details['validation_groups']:,}"
        )
    print(f"[saved] {args.output_manifest}")
    print(f"[saved] {args.output_summary}")


if __name__ == "__main__":
    main()
