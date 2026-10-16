"""Post-process saved prediction probabilities for reviewer-facing metrics.

Supports both CV files (``prob::Class`` columns) and the legacy model-specific
evaluation files (``prob_Class`` columns). No checkpoint or retraining is
required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation_metrics import (
    cluster_bootstrap_ci,
    compute_metric_bundle,
    paired_cluster_bootstrap,
    save_bundle_artifacts,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _load_group_map(fold_manifest: Path | None) -> dict[str, str]:
    if fold_manifest is None:
        return {}
    mapping = {}
    with Path(fold_manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[row["path"]] = row["group_id"]
    return mapping


def _probability_columns(fieldnames: Sequence[str]) -> tuple[list[str], list[str]]:
    cv_columns = [name for name in fieldnames if name.startswith("prob::")]
    if cv_columns:
        return cv_columns, [name.split("::", 1)[1] for name in cv_columns]
    legacy_columns = [name for name in fieldnames if name.startswith("prob_")]
    if legacy_columns:
        return legacy_columns, [name[len("prob_") :] for name in legacy_columns]
    raise ValueError("No probability columns with prefix 'prob::' or 'prob_' found")


def _relative_key(row: dict[str, str], class_name: str) -> str:
    path = Path(row["path"])
    direct = row["path"].replace("\\", "/")
    if not path.is_absolute() and direct.count("/") == 1:
        return direct
    return f"{class_name}/{path.name}"


def load_predictions(
    prediction_path: Path,
    fold_manifest: Path | None = None,
) -> dict[str, Any]:
    group_map = _load_group_map(fold_manifest)
    with Path(prediction_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        probability_columns, class_names = _probability_columns(fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty prediction CSV: {prediction_path}")
    label_column = "label" if "label" in fieldnames else "true_label"
    class_column = "class" if "class" in fieldnames else "true_class"
    if label_column not in fieldnames:
        raise ValueError("Prediction CSV has no label/true_label column")

    labels = np.asarray([int(row[label_column]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [
            [float(row[column]) for column in probability_columns]
            for row in rows
        ],
        dtype=np.float64,
    )
    paths = [row["path"] for row in rows]
    groups = []
    missing_groups = 0
    for row, label in zip(rows, labels, strict=True):
        group = row.get("group_id", "").strip()
        if not group and group_map:
            class_name = (
                row.get(class_column, "").strip()
                if class_column in row
                else class_names[int(label)]
            )
            group = group_map.get(_relative_key(row, class_name), "")
        if not group:
            missing_groups += 1
            group = f"unmapped-singleton:{row['path']}"
        groups.append(group)
    return {
        "rows": rows,
        "paths": paths,
        "labels": labels,
        "probabilities": probabilities,
        "groups": np.asarray(groups, dtype=object),
        "class_names": class_names,
        "missing_group_mappings": missing_groups,
    }


def _align_pair(first: dict[str, Any], second: dict[str, Any]):
    if first["class_names"] != second["class_names"]:
        raise ValueError("Paired files have different probability class columns")
    first_order = np.argsort(np.asarray(first["paths"], dtype=object))
    second_order = np.argsort(np.asarray(second["paths"], dtype=object))
    first_paths = np.asarray(first["paths"], dtype=object)[first_order]
    second_paths = np.asarray(second["paths"], dtype=object)[second_order]
    if not np.array_equal(first_paths, second_paths):
        raise ValueError("Paired prediction files do not contain identical paths")
    labels_a = first["labels"][first_order]
    labels_b = second["labels"][second_order]
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Paired prediction files have different targets")
    groups_a = first["groups"][first_order]
    groups_b = second["groups"][second_order]
    if not np.array_equal(groups_a, groups_b):
        raise ValueError("Paired prediction files have different visual groups")
    return (
        labels_a,
        first["probabilities"][first_order],
        second["probabilities"][second_order],
        groups_a,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--compare-predictions", type=Path, default=None)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path(__file__).resolve().parent
        / "cv_splits"
        / "stratified_group_5fold.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("prediction_analysis"))
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.calibration_bins < 2:
        parser.error("--calibration-bins must be at least 2")
    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations cannot be negative")
    if args.compare_predictions is not None and args.bootstrap_iterations < 1:
        parser.error(
            "--compare-predictions requires a positive --bootstrap-iterations"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prefix = args.prefix or args.predictions.stem
    data = load_predictions(args.predictions, args.fold_manifest)
    bundle = compute_metric_bundle(
        data["labels"],
        data["probabilities"],
        data["class_names"],
        n_bins=args.calibration_bins,
    )
    artifacts = save_bundle_artifacts(
        bundle,
        args.output_dir,
        prefix,
        prefix,
    )
    summary: dict[str, Any] = {
        "predictions": str(args.predictions.resolve()),
        "images": len(data["labels"]),
        "groups": len(set(data["groups"])),
        "missing_group_mappings": data["missing_group_mappings"],
        "metrics": bundle["metrics"],
        "artifacts": artifacts,
    }
    if args.bootstrap_iterations > 0:
        summary["cluster_bootstrap"] = cluster_bootstrap_ci(
            data["labels"],
            data["probabilities"],
            data["groups"],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            n_bins=args.calibration_bins,
        )
    if args.compare_predictions is not None:
        comparison = load_predictions(args.compare_predictions, args.fold_manifest)
        labels, probabilities_a, probabilities_b, groups = _align_pair(
            data,
            comparison,
        )
        summary["paired_comparison"] = paired_cluster_bootstrap(
            labels,
            probabilities_a,
            probabilities_b,
            groups,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            n_bins=args.calibration_bins,
        )
        summary["compare_predictions"] = str(args.compare_predictions.resolve())
    output_path = args.output_dir / f"{prefix}_analysis_summary.json"
    _write_json(output_path, summary)
    print(f"[analysis] metrics={bundle['metrics']}")
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
