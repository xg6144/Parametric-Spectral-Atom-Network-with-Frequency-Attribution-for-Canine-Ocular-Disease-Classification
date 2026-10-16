import sys
import os
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import CanineOcularDataset
from evaluation_metrics import add_reviewer_metrics
from models import psan as M


@dataclass
class EvalConfig:
    gpu_id: int = 0
    data_dir: str = "dataset_split_dedup_phash0"
    split: str = "test"
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    seed: int = 42
    num_classes: int = 11

    ckpt_path: str = "best_psan_ti.pth"
    weight_source: str = "auto"  # "auto" | "eval_model" | "ema" | "model"

    output_dir: str = "eval_results"
    model_tag: str = "psan_ti"

    bootstrap: bool = True
    n_bootstrap: int = 1000

    measure_latency: bool = True
    latency_warmup: int = 20
    latency_iters: int = 100
    latency_batch_size: int = 1


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_PSAN_BUILDERS = {
    "psan_ti": M.psan_ti,
    "psan_xs": M.psan_xs,
    "psan_s":  M.psan_s,
    "psan_b":  M.psan_b,
}


def build_model_from_ckpt_config(
    ckpt_cfg: Dict[str, Any], num_classes: int, img_size: int,
) -> nn.Module:
    model_name = ckpt_cfg.get("model") or "psan_ti"
    if model_name not in _PSAN_BUILDERS:
        raise ValueError(
            f"Unknown PSAN variant: {model_name}. "
            f"Supported: {list(_PSAN_BUILDERS)}."
        )
    factory = _PSAN_BUILDERS[model_name]

    psan_kwargs = dict(
        M=ckpt_cfg.get("atom_count", 16),
        atom=ckpt_cfg.get("atom", "gabor"),
        init=ckpt_cfg.get("init", "morlet"),
        anisotropic=not ckpt_cfg.get("isotropic", False),
        learn_phase=not ckpt_cfg.get("no_phase", False),
        atom_dropout=0.0,
    )

    model = factory(
        num_classes=num_classes,
        psan_kwargs=psan_kwargs,
        img_size=ckpt_cfg.get("img_size", img_size),
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    return model


def load_state_for_eval(model: nn.Module, ckpt: Dict[str, Any], weight_source: str) -> str:
    if weight_source == "auto":
        for k in ("eval_model", "ema", "model"):
            if ckpt.get(k) is not None:
                chosen = k
                break
        else:
            raise KeyError("checkpoint에 사용 가능한 weight 키가 없습니다.")
    else:
        if ckpt.get(weight_source) is None:
            raise KeyError(f"weight_source='{weight_source}' 키가 checkpoint에 없습니다.")
        chosen = weight_source
      
    sd = {
        k: v for k, v in ckpt[chosen].items()
        if not (k.endswith(".filter.U") or k.endswith(".filter.V"))
    }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Warn] missing keys: {len(missing)} (first5={missing[:5]})")
    if unexpected:
        print(f"[Warn] unexpected keys: {len(unexpected)} (first5={unexpected[:5]})")
    return chosen


@torch.no_grad()
def run_inference(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()

    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(y.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    return {"logits": logits, "probs": probs, "preds": preds, "labels": labels}


def compute_full_metrics(labels, preds, probs, num_classes, class_names) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    m["accuracy"] = float(accuracy_score(labels, preds))
    m["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    m["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    m["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    m["macro_precision"] = float(precision_score(labels, preds, average="macro", zero_division=0))
    m["macro_recall"] = float(recall_score(labels, preds, average="macro", zero_division=0))
    m["weighted_precision"] = float(precision_score(labels, preds, average="weighted", zero_division=0))
    m["weighted_recall"] = float(recall_score(labels, preds, average="weighted", zero_division=0))
    m["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    m["mcc"] = float(matthews_corrcoef(labels, preds))

    try:
        if num_classes > 2:
            m["macro_auc_ovr"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
            m["weighted_auc_ovr"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
            )
            m["macro_auc_ovo"] = float(
                roc_auc_score(labels, probs, multi_class="ovo", average="macro")
            )
        else:
            m["auc"] = float(roc_auc_score(labels, probs[:, 1]))
    except Exception as e:
        print(f"[Warn] AUC 계산 실패: {e}")

    per_class_auc = {}
    try:
        if num_classes > 2:
            y_bin = label_binarize(labels, classes=list(range(num_classes)))
            for i, cls_name in enumerate(class_names):
                try:
                    per_class_auc[cls_name] = float(
                        roc_auc_score(y_bin[:, i], probs[:, i])
                    )
                except ValueError:
                    per_class_auc[cls_name] = float("nan")
        else:
            per_class_auc[class_names[0]] = float(
                roc_auc_score(labels, probs[:, 0], pos_label=0)
            )
            per_class_auc[class_names[1]] = float(
                roc_auc_score(labels, probs[:, 1])
            )
    except Exception as e:
        print(f"[Warn] per-class AUC 계산 실패: {e}")
    m["per_class_auc"] = per_class_auc

    m["per_class_report"] = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return m


def bootstrap_ci(labels, preds, probs, num_classes, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
    auc_key = "macro_auc_ovr" if num_classes > 2 else "auc"
    keys.append(auc_key)
    boot = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        l, p, pr = labels[idx], preds[idx], probs[idx]
        boot["accuracy"].append(accuracy_score(l, p))
        try:
            boot["balanced_accuracy"].append(balanced_accuracy_score(l, p))
        except Exception:
            boot["balanced_accuracy"].append(np.nan)
        boot["macro_f1"].append(f1_score(l, p, average="macro", zero_division=0))
        boot["weighted_f1"].append(f1_score(l, p, average="weighted", zero_division=0))
        boot["cohen_kappa"].append(cohen_kappa_score(l, p))
        try:
            if num_classes > 2:
                if len(np.unique(l)) == num_classes:
                    boot[auc_key].append(
                        roc_auc_score(l, pr, multi_class="ovr", average="macro")
                    )
                else:
                    boot[auc_key].append(np.nan)
            else:
                boot[auc_key].append(roc_auc_score(l, pr[:, 1]))
        except Exception:
            boot[auc_key].append(np.nan)

    summary = {}
    for k, v in boot.items():
        arr = np.array(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            summary[k] = {"mean": float("nan"), "ci95_lower": float("nan"),
                          "ci95_upper": float("nan"), "std": float("nan")}
            continue
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95_lower": float(np.percentile(arr, 2.5)),
            "ci95_upper": float(np.percentile(arr, 97.5)),
        }
    return summary


def save_confusion_matrix(labels, preds, class_names, out_png, out_csv):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_csv)
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        out_csv.replace(".csv", "_normalized.csv")
    )

    side = max(6, len(class_names) * 0.7)
    fig, axes = plt.subplots(1, 2, figsize=(2 * side, side))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalized)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def measure_flops(model, device, img_size):
    from model_resources import measure_supported_operations
    return measure_supported_operations(model, device, img_size)


@torch.no_grad()
def measure_latency(model, device, img_size, warmup, iters, batch_size):
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "batch_size": batch_size,
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "p50_ms": float(np.percentile(times, 50) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "throughput_imgs_per_sec": float(batch_size / times.mean()),
    }


def main(cfg: EvalConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)
        device = torch.device(f"cuda:{cfg.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    print(f"[Ckpt] loading {cfg.ckpt_path}")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    print(f"[Ckpt] epoch={ckpt.get('epoch','?')} "
          f"best_macro_f1(train)={ckpt.get('best_macro_f1','?')}")

    model = build_model_from_ckpt_config(
        ckpt_cfg, num_classes=cfg.num_classes, img_size=cfg.img_size,
    )
    chosen = load_state_for_eval(model, ckpt, cfg.weight_source)
    print(f"[Ckpt] loaded weights key='{chosen}'")
    model.to(device).eval()

    model_name = ckpt_cfg.get("model") or "psan_ti"
    pc = M.count_params(model)
    print(
        f"[Model] {model_name} "
        f"total_params={pc['total']:,} ({pc['total']/1e6:.2f}M)  "
        f"filter_only={pc['filter_only']:,}  "
        f"atom_count={ckpt_cfg.get('atom_count','?')}  "
        f"atom={ckpt_cfg.get('atom','?')}  init={ckpt_cfg.get('init','?')}  "
        f"isotropic={ckpt_cfg.get('isotropic','?')}  "
        f"no_phase={ckpt_cfg.get('no_phase','?')}  "
        f"drop_rate={ckpt_cfg.get('drop_rate','?')}  "
        f"drop_path_rate={ckpt_cfg.get('drop_path_rate','?')}"
    )

    flops_info = measure_flops(model, device, cfg.img_size)
    if flops_info is not None:
        print(f"[Model] FLOPs={flops_info['flops_G']:.2f}G  "
              f"MACs={flops_info['macs_G']:.2f}G")

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    test_ds = CanineOcularDataset(
        cfg.data_dir, split=cfg.split, img_size=cfg.img_size, transform=tfm,
    )
    if len(test_ds) == 0:
        raise RuntimeError(f"split='{cfg.split}'에서 샘플을 찾지 못했습니다: {cfg.data_dir}")

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    class_names = test_ds.classes
    print(f"[Data] split={cfg.split} N={len(test_ds)} classes={len(class_names)}")

    t0 = time.time()
    out = run_inference(model, test_loader, device)
    t1 = time.time() - t0
    print(f"[Inference] N={len(out['labels'])} elapsed={t1:.2f}s "
          f"({len(out['labels'])/t1:.1f} img/s incl. dataloader)")

    metrics = compute_full_metrics(
        out["labels"], out["preds"], out["probs"],
        num_classes=cfg.num_classes, class_names=class_names,
    )
    add_reviewer_metrics(
        metrics, out["labels"], out["probs"], class_names,
        cfg.output_dir, f"{cfg.model_tag}_reviewer", cfg.model_tag,
    )

    print("\n========== TEST METRICS ==========")
    for k in ["accuracy", "balanced_accuracy",
              "macro_f1", "weighted_f1",
              "macro_precision", "macro_recall",
              "macro_auc_ovr", "weighted_auc_ovr", "macro_auc_ovo",
              "cohen_kappa", "mcc"]:
        if k in metrics:
            print(f"  {k:<22s}: {metrics[k]:.4f}")

    if metrics.get("per_class_auc"):
        print("\n---------- Per-class AUC (OvR) ----------")
        for cls_name, auc_val in metrics["per_class_auc"].items():
            print(f"  {cls_name:<22s}: {auc_val:.4f}")

    if cfg.bootstrap:
        print(f"\n[Bootstrap] computing 95% CI (n={cfg.n_bootstrap}) ...")
        ci = bootstrap_ci(
            out["labels"], out["preds"], out["probs"],
            num_classes=cfg.num_classes,
            n_bootstrap=cfg.n_bootstrap, seed=cfg.seed,
        )
        print("\n========== 95% CI (bootstrap) ==========")
        for k, v in ci.items():
            print(f"  {k:<22s}: {v['mean']:.4f} "
                  f"[{v['ci95_lower']:.4f}, {v['ci95_upper']:.4f}]")
        metrics["bootstrap_ci"] = ci

    cm_png = os.path.join(cfg.output_dir, f"{cfg.model_tag}_confusion_matrix.png")
    cm_csv = os.path.join(cfg.output_dir, f"{cfg.model_tag}_confusion_matrix.csv")
    save_confusion_matrix(out["labels"], out["preds"], class_names, cm_png, cm_csv)
    print(f"[Saved] {cm_png}")
    print(f"[Saved] {cm_csv}")

    per_class_auc = metrics.get("per_class_auc", {})
    rows = []
    for cls_name in class_names:
        r = metrics["per_class_report"].get(cls_name)
        if r is None:
            continue
        rows.append({
            "class": cls_name,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "auc_ovr": per_class_auc.get(cls_name, float("nan")),
            "support": int(r["support"]),
        })
    per_csv = os.path.join(cfg.output_dir, f"{cfg.model_tag}_per_class_metrics.csv")
    pd.DataFrame(rows).to_csv(per_csv, index=False)
    print(f"[Saved] {per_csv}")

    pred_rows = []
    for i, (path, y_true) in enumerate(test_ds.samples):
        row = {
            "path": path,
            "true_label": int(y_true),
            "true_class": class_names[int(y_true)],
            "pred_label": int(out["preds"][i]),
            "pred_class": class_names[int(out["preds"][i])],
            "correct": bool(int(y_true) == int(out["preds"][i])),
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(out["probs"][i, c_idx])
        pred_rows.append(row)
    pred_csv = os.path.join(cfg.output_dir, f"{cfg.model_tag}_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)
    print(f"[Saved] {pred_csv}")

    if cfg.measure_latency:
        print(f"\n[Latency] warmup={cfg.latency_warmup} iters={cfg.latency_iters} "
              f"bs={cfg.latency_batch_size}")
        lat = measure_latency(
            model, device, cfg.img_size,
            warmup=cfg.latency_warmup, iters=cfg.latency_iters,
            batch_size=cfg.latency_batch_size,
        )
        for k, v in lat.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        metrics["latency"] = lat

    _ad_train = ckpt_cfg.get("atom_dropout")

    summary = {
        "config": asdict(cfg),
        "ckpt_info": {
            "path": cfg.ckpt_path,
            "epoch": ckpt.get("epoch"),
            "best_macro_f1_at_train": ckpt.get("best_macro_f1"),
            "weight_source": chosen,
        },
        "model_info": {
            "model": model_name,
            "atom_count": ckpt_cfg.get("atom_count"),
            "atom": ckpt_cfg.get("atom"),
            "init": ckpt_cfg.get("init"),
            "isotropic": ckpt_cfg.get("isotropic"),
            "no_phase": ckpt_cfg.get("no_phase"),
            "atom_dropout_train": _ad_train,
            "drop_rate_train": ckpt_cfg.get("drop_rate"),
            "drop_path_rate_train": ckpt_cfg.get("drop_path_rate"),
            "total_params": pc["total"],
            "total_params_M": pc["total"] / 1e6,
            "filter_only_params": pc["filter_only"],
            **(flops_info if flops_info is not None else {}),
        },
        "data_info": {
            "split": cfg.split,
            "n_samples": len(test_ds),
            "classes": class_names,
        },
        "metrics": metrics,
    }
    json_path = os.path.join(cfg.output_dir, f"{cfg.model_tag}_eval_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Saved] {json_path}")

    print("\n[Done] evaluation complete.")

if __name__ == "__main__":
    cfg = EvalConfig(
        gpu_id=1,
        data_dir="dataset_split_dedup_phash0",
        split="test",
        img_size=224,
        batch_size=64,
        num_workers=8,
        seed=42,
        num_classes=11,

        ckpt_path="train_results/psan_ti_M16_gabor_morlet_bw02/20260605_121030/best_psan_ti_M16_gabor_morlet_bw02.pth",
        weight_source="auto",

        output_dir="eval_results",
        model_tag="psan_ti_M16_gabor_morlet_ad02",

        bootstrap=True,
        n_bootstrap=1000,

        measure_latency=True,
        latency_warmup=20,
        latency_iters=100,
        latency_batch_size=1,
    )
    main(cfg)
