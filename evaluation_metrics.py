"""Shared classification, calibration, and group-aware uncertainty metrics.

Definitions
-----------
* ECE is top-label expected calibration error with equal-width bins.
* Classwise ECE treats every class as a one-vs-rest calibration problem and
  reports the unweighted mean across classes.
* Multiclass Brier score is ``mean(sum_c (p_c - y_c)^2)`` (not divided by the
  number of classes).
* PR-AUC is average precision, which is the standard step-wise area under the
  precision-recall curve.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


EPSILON = 1e-12
DEFAULT_BOOTSTRAP_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "ece",
    "brier_score",
    "nll",
    "macro_pr_auc",
)
LOWER_IS_BETTER = {"ece", "classwise_ece", "brier_score", "nll"}


def _validate_inputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {labels.shape}")
    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must have shape (N, C), got {probabilities.shape}"
        )
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("labels and probabilities have different lengths")
    if probabilities.shape[1] < 2:
        raise ValueError("At least two probability columns are required")
    if labels.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("A label is outside the probability column range")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinity")
    if np.any(probabilities < -1e-7):
        raise ValueError("Probabilities contain negative values")

    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("A probability row sums to zero")
    # Normalize small serialization/rounding drift while rejecting logits.
    maximum_drift = float(np.max(np.abs(row_sums[:, 0] - 1.0)))
    if maximum_drift > 1e-3:
        raise ValueError(
            "Probability rows do not sum to one; maximum drift is "
            f"{maximum_drift:.6g}"
        )
    probabilities = probabilities / row_sums
    return labels, probabilities


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def _balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Mean recall over true classes with support, without sklearn warnings."""

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(num_classes),
        sample_weight=sample_weight,
    )
    support = matrix.sum(axis=1)
    present = support > 0
    if not np.any(present):
        return float("nan")
    recall = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=present,
    )
    return float(np.mean(recall[present]))


def calibration_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Return equal-width top-label reliability bins."""

    labels, probabilities = _validate_inputs(labels, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != labels.shape:
        raise ValueError("sample_weight has the wrong shape")
    bin_indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)

    result: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        bin_weight = float(weights[mask].sum())
        if bin_weight > 0.0:
            average_confidence = float(
                np.sum(confidence[mask] * weights[mask]) / bin_weight
            )
            average_accuracy = float(
                np.sum(correct[mask] * weights[mask]) / bin_weight
            )
        else:
            average_confidence = float("nan")
            average_accuracy = float("nan")
        result.append(
            {
                "bin": bin_index + 1,
                "lower": bin_index / n_bins,
                "upper": (bin_index + 1) / n_bins,
                "count": int(mask.sum()),
                "weight": bin_weight,
                "average_confidence": average_confidence,
                "average_accuracy": average_accuracy,
                "gap": (
                    abs(average_accuracy - average_confidence)
                    if bin_weight > 0.0
                    else float("nan")
                ),
            }
        )
    return result


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> float:
    bins = calibration_bins(labels, probabilities, n_bins, sample_weight)
    total_weight = sum(float(item["weight"]) for item in bins)
    if total_weight <= 0.0:
        return float("nan")
    return float(
        sum(
            float(item["weight"]) * float(item["gap"])
            for item in bins
            if float(item["weight"]) > 0.0
        )
        / total_weight
    )


def binary_expected_calibration_error(
    targets: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    weights = (
        np.ones(len(targets), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    bin_indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return float("nan")
    error = 0.0
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        bin_weight = float(weights[mask].sum())
        if bin_weight <= 0.0:
            continue
        observed = float(np.sum(targets[mask] * weights[mask]) / bin_weight)
        predicted = float(np.sum(confidence[mask] * weights[mask]) / bin_weight)
        error += bin_weight / total_weight * abs(observed - predicted)
    return float(error)


def classwise_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    n_bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    labels, probabilities = _validate_inputs(labels, probabilities)
    values = {
        name: binary_expected_calibration_error(
            labels == class_index,
            probabilities[:, class_index],
            n_bins,
            sample_weight,
        )
        for class_index, name in enumerate(class_names)
    }
    return float(np.nanmean(list(values.values()))), values


def multiclass_brier_score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    labels, probabilities = _validate_inputs(labels, probabilities)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    per_image = np.sum((probabilities - one_hot) ** 2, axis=1)
    return _weighted_mean(per_image, sample_weight)


def negative_log_likelihood(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    labels, probabilities = _validate_inputs(labels, probabilities)
    per_image = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], EPSILON, 1.0))
    return _weighted_mean(per_image, sample_weight)


def compute_metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute global, per-class, calibration, and confusion outputs."""

    labels, probabilities = _validate_inputs(labels, probabilities)
    class_names = list(class_names)
    if len(class_names) != probabilities.shape[1]:
        raise ValueError("class_names does not match probability columns")
    num_classes = len(class_names)
    predictions = probabilities.argmax(axis=1)
    class_indices = np.arange(num_classes)
    y_binary = label_binarize(labels, classes=class_indices)
    if num_classes == 2 and y_binary.shape[1] == 1:
        y_binary = np.column_stack([1 - y_binary[:, 0], y_binary[:, 0]])

    global_metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": _balanced_accuracy(
            labels,
            predictions,
            num_classes,
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                labels,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                labels,
                predictions,
                labels=class_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "ece": expected_calibration_error(labels, probabilities, n_bins),
        "brier_score": multiclass_brier_score(labels, probabilities),
        "nll": negative_log_likelihood(labels, probabilities),
    }
    classwise_ece, per_class_ece = classwise_calibration(
        labels,
        probabilities,
        class_names,
        n_bins,
    )
    global_metrics["classwise_ece"] = classwise_ece

    all_classes_present = len(np.unique(labels)) == num_classes
    try:
        if not all_classes_present:
            raise ValueError("At least one class is absent")
        global_metrics["macro_auc_ovr"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
        global_metrics["weighted_auc_ovr"] = float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )
        )
        global_metrics["macro_auc_ovo"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovo", average="macro")
        )
    except ValueError:
        global_metrics["macro_auc_ovr"] = float("nan")
        global_metrics["weighted_auc_ovr"] = float("nan")
        global_metrics["macro_auc_ovo"] = float("nan")

    try:
        if not all_classes_present:
            raise ValueError("At least one class is absent")
        global_metrics["macro_pr_auc"] = float(
            average_precision_score(y_binary, probabilities, average="macro")
        )
        global_metrics["weighted_pr_auc"] = float(
            average_precision_score(y_binary, probabilities, average="weighted")
        )
    except ValueError:
        global_metrics["macro_pr_auc"] = float("nan")
        global_metrics["weighted_pr_auc"] = float("nan")

    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    normalized = matrix.astype(np.float64) / np.maximum(
        matrix.sum(axis=1, keepdims=True),
        1,
    )
    per_class: list[dict[str, float | int | str]] = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(matrix[class_index, class_index])
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_negative - false_positive)
        support = int(np.sum(labels == class_index))
        sensitivity = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else float("nan")
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else float("nan")
        )
        binary_targets = y_binary[:, class_index]
        if np.unique(binary_targets).size < 2:
            auc = float("nan")
            pr_auc = float("nan")
        else:
            auc = float(
                roc_auc_score(binary_targets, probabilities[:, class_index])
            )
            pr_auc = float(
                average_precision_score(binary_targets, probabilities[:, class_index])
            )
        per_class.append(
            {
                "class": class_name,
                "class_index": class_index,
                "support": support,
                "precision": float(
                    precision_score(
                        labels == class_index,
                        predictions == class_index,
                        zero_division=0,
                    )
                ),
                "sensitivity": float(sensitivity),
                "recall": float(sensitivity),
                "specificity": float(specificity),
                "f1": float(
                    f1_score(
                        labels == class_index,
                        predictions == class_index,
                        zero_division=0,
                    )
                ),
                "roc_auc_ovr": auc,
                "pr_auc": pr_auc,
                "ece_ovr": per_class_ece[class_name],
                "tp": true_positive,
                "fn": false_negative,
                "fp": false_positive,
                "tn": true_negative,
            }
        )

    return {
        "definitions": {
            "ece": f"top-label ECE with {n_bins} equal-width bins",
            "classwise_ece": (
                f"mean one-vs-rest ECE with {n_bins} equal-width bins"
            ),
            "brier_score": "mean(sum_c((p_c-y_c)^2)); not divided by C",
            "pr_auc": "average precision",
        },
        "metrics": global_metrics,
        "per_class": per_class,
        "calibration_bins": calibration_bins(labels, probabilities, n_bins),
        "per_class_ece": per_class_ece,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
    }


def _selected_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray,
    metric_names: Sequence[str],
    n_bins: int,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    selected: dict[str, float] = {}
    if "macro_f1" in metric_names:
        selected["macro_f1"] = float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(probabilities.shape[1]),
                average="macro",
                zero_division=0,
                sample_weight=sample_weight,
            )
        )
    if "balanced_accuracy" in metric_names:
        selected["balanced_accuracy"] = _balanced_accuracy(
            labels,
            predictions,
            probabilities.shape[1],
            sample_weight,
        )
    if "ece" in metric_names:
        selected["ece"] = expected_calibration_error(
            labels,
            probabilities,
            n_bins,
            sample_weight,
        )
    if "brier_score" in metric_names:
        selected["brier_score"] = multiclass_brier_score(
            labels,
            probabilities,
            sample_weight,
        )
    if "nll" in metric_names:
        selected["nll"] = negative_log_likelihood(
            labels,
            probabilities,
            sample_weight,
        )
    if "macro_pr_auc" in metric_names:
        y_binary = label_binarize(
            labels,
            classes=np.arange(probabilities.shape[1]),
        )
        if probabilities.shape[1] == 2 and y_binary.shape[1] == 1:
            y_binary = np.column_stack([1 - y_binary[:, 0], y_binary[:, 0]])
        weighted_support = np.asarray(
            [
                np.sum(sample_weight[labels == class_index])
                for class_index in range(probabilities.shape[1])
            ]
        )
        if np.any(weighted_support <= 0.0):
            selected["macro_pr_auc"] = float("nan")
        else:
            selected["macro_pr_auc"] = float(
                average_precision_score(
                    y_binary,
                    probabilities,
                    average="macro",
                    sample_weight=sample_weight,
                )
            )
    unknown = set(metric_names) - set(selected)
    if unknown:
        raise ValueError(f"Unsupported bootstrap metrics: {sorted(unknown)}")
    return selected


def _group_codes(groups: Sequence[str], expected_length: int) -> tuple[np.ndarray, int]:
    groups = np.asarray(groups, dtype=object)
    if groups.shape != (expected_length,):
        raise ValueError("groups has the wrong shape")
    _, codes = np.unique(groups, return_inverse=True)
    return codes, int(codes.max()) + 1


def cluster_bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: Sequence[str],
    iterations: int = 1_000,
    seed: int = 42,
    metric_names: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Percentile CI from a visual-group cluster bootstrap.

    Groups are sampled with replacement. All images in a sampled group receive
    the same multiplicity through ``sample_weight``.
    """

    labels, probabilities = _validate_inputs(labels, probabilities)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    codes, group_count = _group_codes(groups, len(labels))
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(iterations, dtype=np.float64) for metric in metric_names}
    for iteration in range(iterations):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        multiplicity = np.bincount(sampled_groups, minlength=group_count)
        sample_weight = multiplicity[codes].astype(np.float64)
        metrics = _selected_metrics(
            labels,
            probabilities,
            sample_weight,
            metric_names,
            n_bins,
        )
        for metric in metric_names:
            values[metric][iteration] = metrics[metric]

    observed = _selected_metrics(
        labels,
        probabilities,
        np.ones(len(labels), dtype=np.float64),
        metric_names,
        n_bins,
    )
    summary = {}
    for metric, distribution in values.items():
        finite = distribution[np.isfinite(distribution)]
        if len(finite):
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            lower = float(np.percentile(finite, 2.5))
            upper = float(np.percentile(finite, 97.5))
        else:
            mean = std = lower = upper = float("nan")
        summary[metric] = {
            "observed": observed[metric],
            "bootstrap_mean": mean,
            "bootstrap_std": std,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "valid_iterations": int(len(finite)),
        }
    return {
        "method": "visual-group cluster percentile bootstrap",
        "iterations": iterations,
        "seed": seed,
        "groups": group_count,
        "metrics": summary,
    }


def paired_cluster_bootstrap(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    groups: Sequence[str],
    iterations: int = 1_000,
    seed: int = 42,
    metric_names: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Paired visual-group bootstrap for A minus B metric differences."""

    labels, probabilities_a = _validate_inputs(labels, probabilities_a)
    labels_b, probabilities_b = _validate_inputs(labels, probabilities_b)
    if not np.array_equal(labels, labels_b):
        raise ValueError("The paired models have different target arrays")
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError("The paired models have different probability shapes")
    codes, group_count = _group_codes(groups, len(labels))
    rng = np.random.default_rng(seed)
    differences = {
        metric: np.empty(iterations, dtype=np.float64) for metric in metric_names
    }
    unit_weight = np.ones(len(labels), dtype=np.float64)
    observed_a = _selected_metrics(
        labels, probabilities_a, unit_weight, metric_names, n_bins
    )
    observed_b = _selected_metrics(
        labels, probabilities_b, unit_weight, metric_names, n_bins
    )
    for iteration in range(iterations):
        sampled_groups = rng.integers(0, group_count, size=group_count)
        multiplicity = np.bincount(sampled_groups, minlength=group_count)
        sample_weight = multiplicity[codes].astype(np.float64)
        metrics_a = _selected_metrics(
            labels, probabilities_a, sample_weight, metric_names, n_bins
        )
        metrics_b = _selected_metrics(
            labels, probabilities_b, sample_weight, metric_names, n_bins
        )
        for metric in metric_names:
            differences[metric][iteration] = metrics_a[metric] - metrics_b[metric]

    summary = {}
    for metric, distribution in differences.items():
        finite = distribution[np.isfinite(distribution)]
        if not len(finite):
            summary[metric] = {
                "observed_a": observed_a[metric],
                "observed_b": observed_b[metric],
                "observed_difference_a_minus_b": (
                    observed_a[metric] - observed_b[metric]
                ),
                "ci95_lower": float("nan"),
                "ci95_upper": float("nan"),
                "two_sided_p": float("nan"),
                "higher_is_better": metric not in LOWER_IS_BETTER,
                "probability_a_better": float("nan"),
                "valid_iterations": 0,
            }
            continue
        probability_nonpositive = (np.sum(finite <= 0.0) + 1) / (len(finite) + 1)
        probability_nonnegative = (np.sum(finite >= 0.0) + 1) / (len(finite) + 1)
        p_value = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        observed_difference = observed_a[metric] - observed_b[metric]
        higher_is_better = metric not in LOWER_IS_BETTER
        oriented = finite if higher_is_better else -finite
        summary[metric] = {
            "observed_a": observed_a[metric],
            "observed_b": observed_b[metric],
            "observed_difference_a_minus_b": observed_difference,
            "ci95_lower": float(np.percentile(finite, 2.5)),
            "ci95_upper": float(np.percentile(finite, 97.5)),
            "two_sided_p": float(p_value),
            "higher_is_better": higher_is_better,
            "probability_a_better": float(np.mean(oriented > 0.0)),
            "valid_iterations": int(len(finite)),
        }
    return {
        "method": "paired visual-group cluster percentile bootstrap",
        "difference": "model_a minus model_b",
        "iterations": iterations,
        "seed": seed,
        "groups": group_count,
        "metrics": summary,
    }


def save_reliability_diagram(
    bundle: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = bundle["calibration_bins"]
    centers = np.asarray(
        [(float(item["lower"]) + float(item["upper"])) / 2.0 for item in bins]
    )
    accuracies = np.asarray([float(item["average_accuracy"]) for item in bins])
    confidences = np.asarray([float(item["average_confidence"]) for item in bins])
    counts = np.asarray([int(item["count"]) for item in bins])
    valid = np.isfinite(accuracies) & np.isfinite(confidences)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    axes[0].plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    axes[0].plot(
        confidences[valid],
        accuracies[valid],
        marker="o",
        linewidth=2,
        label="Observed",
    )
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy")
    axes[0].set_title(
        f"{title}\nECE={bundle['metrics']['ece']:.4f}, "
        f"Brier={bundle['metrics']['brier_score']:.4f}, "
        f"NLL={bundle['metrics']['nll']:.4f}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    width = 1.0 / len(bins)
    axes[1].bar(centers, counts, width=width * 0.9, color="#4C72B0")
    axes[1].set(xlim=(0, 1), xlabel="Confidence", ylabel="Count")
    axes[1].grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(temporary, dpi=300, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, output_path)


def save_confusion_matrix_figure(
    bundle: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    """Save count and row-normalized confusion matrices in one 300-dpi PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = [row["class"] for row in bundle["per_class"]]
    matrices = (
        (np.asarray(bundle["confusion_matrix"]), "Count", "d"),
        (
            np.asarray(bundle["confusion_matrix_normalized"], dtype=np.float64),
            "Row normalized",
            ".2f",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for axis, (matrix, subtitle, value_format) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set(
            title=subtitle,
            xlabel="Predicted class",
            ylabel="True class",
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axis.tick_params(axis="x", labelrotation=45)
        threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
    figure.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(temporary, dpi=300, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, output_path)


def save_bundle_artifacts(
    bundle: dict[str, Any],
    output_dir: Path,
    prefix: str,
    title: str,
) -> dict[str, str]:
    """Save JSON, per-class CSV, confusion matrices, and reliability plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}_evaluation.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_json, json_path)

    per_class_path = output_dir / f"{prefix}_per_class_metrics.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        rows = bundle["per_class"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class_names = [row["class"] for row in bundle["per_class"]]
    matrix_paths = {}
    for key, suffix in (
        ("confusion_matrix", "confusion_matrix_count"),
        ("confusion_matrix_normalized", "confusion_matrix_normalized"),
    ):
        matrix_path = output_dir / f"{prefix}_{suffix}.csv"
        matrix = np.asarray(bundle[key])
        with matrix_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true/pred", *class_names])
            for class_name, values in zip(class_names, matrix, strict=True):
                writer.writerow([class_name, *values.tolist()])
        matrix_paths[suffix] = str(matrix_path.resolve())

    reliability_path = output_dir / f"{prefix}_reliability.png"
    save_reliability_diagram(bundle, reliability_path, title)
    confusion_figure_path = output_dir / f"{prefix}_confusion_matrix.png"
    save_confusion_matrix_figure(bundle, confusion_figure_path, title)
    return {
        "evaluation_json": str(json_path.resolve()),
        "per_class_csv": str(per_class_path.resolve()),
        "reliability_plot": str(reliability_path.resolve()),
        "confusion_matrix_plot": str(confusion_figure_path.resolve()),
        **matrix_paths,
    }


def add_reviewer_metrics(
    metrics: dict[str, Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path | str,
    prefix: str,
    title: str,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Augment a legacy evaluation dictionary without breaking its schema.

    The model-specific evaluators predate the reviewer-requested calibration
    and per-class metrics.  This adapter keeps their existing keys and files,
    while adding a complete nested evaluation and separate reviewer artifacts.
    """

    bundle = compute_metric_bundle(
        labels,
        probabilities,
        class_names,
        n_bins=n_bins,
    )
    metrics.update(bundle["metrics"])
    metrics["reviewer_evaluation"] = bundle
    metrics["reviewer_artifacts"] = save_bundle_artifacts(
        bundle,
        Path(output_dir),
        prefix,
        title,
    )
    return bundle
