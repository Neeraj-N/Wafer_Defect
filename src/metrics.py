"""
Metrics that matter for a fab, beyond raw accuracy.

Three things live here:

  * `classification_metrics` -- per-class precision / recall / F1 / support
    plus macro-F1, balanced accuracy and (for reference) raw accuracy. On this
    dataset raw accuracy is misleading because ~85% of wafers are `none`, so a
    do-nothing classifier scores ~0.85; macro-F1 and per-class recall are what
    tell you whether rare patterns are actually caught.

  * `cost_weighted_error` -- average misclassification cost under an explicit
    cost matrix. In inspection, errors are not symmetric: shipping a defective
    wafer as `none` (an escape) is far more expensive than a false alarm that
    triggers a re-check. The default matrix here is a documented *placeholder*
    shaped like that asymmetry; a real deployment tunes it to its own scrap and
    re-inspection costs.

  * `benchmark_latency` -- measured single-wafer inference latency (ms) and
    batched throughput (wafers/s), with warm-up. These are honest measurements
    of whatever machine you run them on; do not copy a number from a README.

`write_report` serialises everything to JSON and to a Markdown table you can
paste into the README.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from .config import FAILURE_CLASSES


def classification_metrics(targets, preds, class_names=FAILURE_CLASSES) -> dict:
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    return {
        "accuracy": float((targets == preds).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        # macro-F1 over the classes actually present, matching how src.train and
        # src.evaluate already compute it (sklearn's average="macro" default), so
        # this reproduces the headline number in the README on the full test set.
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def default_cost_matrix(class_names=FAILURE_CLASSES, none_class: str = "none") -> np.ndarray:
    """A documented, illustrative asymmetric cost matrix C where C[t, p] is the
    cost of predicting `p` when the truth is `t`:

      * correct prediction                       -> 0
      * escape (a real defect called `none`)     -> 5  (worst: a bad wafer ships)
      * false alarm (`none` called a defect)     -> 1  (a needless re-check)
      * defect-for-different-defect confusion    -> 2  (misrouted, but caught)

    Replace these with your fab's real relative costs before quoting the number.
    """
    n = len(class_names)
    none_idx = class_names.index(none_class)
    c = np.full((n, n), 2.0)
    np.fill_diagonal(c, 0.0)
    c[none_idx, :] = 1.0          # truth is none, predicted a defect -> false alarm
    c[:, none_idx] = 5.0          # truth is a defect, predicted none -> escape
    c[none_idx, none_idx] = 0.0
    return c


def cost_weighted_error(targets, preds, cost_matrix=None, class_names=FAILURE_CLASSES) -> float:
    """Mean cost per wafer under `cost_matrix` (defaults to default_cost_matrix)."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    c = default_cost_matrix(class_names) if cost_matrix is None else np.asarray(cost_matrix)
    return float(c[targets, preds].mean())


def benchmark_latency(model, input_shape=(2, 64, 64), device="cpu",
                      n_warmup: int = 5, n_iter: int = 50, batch_size: int = 128) -> dict:
    """Measure single-wafer latency and batched throughput. Returns a dict with
    `latency_ms_per_wafer` (batch-1) and `throughput_wafers_per_s` (batched)."""
    import torch

    model = model.to(device).eval()
    x1 = torch.randn(1, *input_shape, device=device)
    xb = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x1)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x1)
        if device == "cuda":
            torch.cuda.synchronize()
        single_ms = (time.perf_counter() - t0) / n_iter * 1000.0

        for _ in range(n_warmup):
            model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        batched_s = (time.perf_counter() - t0) / n_iter
    return {
        "device": str(device),
        "batch_size": batch_size,
        "latency_ms_per_wafer": round(single_ms, 3),
        "throughput_wafers_per_s": round(batch_size / batched_s, 1),
    }


def to_markdown(metrics: dict, latency: dict | None = None,
                cost: float | None = None) -> str:
    lines = [
        "| Metric | Score |",
        "| --- | --- |",
        f"| Macro-F1 | {metrics['macro_f1']:.3f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.3f} |",
        f"| Raw accuracy | {metrics['accuracy']:.3f} |",
    ]
    if cost is not None:
        lines.append(f"| Cost-weighted error (illustrative matrix) | {cost:.3f} |")
    if latency is not None:
        lines.append(
            f"| Inference latency ({latency['device']}, batch 1) | "
            f"{latency['latency_ms_per_wafer']:.2f} ms/wafer |"
        )
        lines.append(
            f"| Throughput ({latency['device']}, batch {latency['batch_size']}) | "
            f"{latency['throughput_wafers_per_s']:.0f} wafers/s |"
        )
    lines += ["", "| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"]
    for cls, d in metrics["per_class"].items():
        lines.append(
            f"| {cls} | {d['precision']:.2f} | {d['recall']:.2f} | {d['f1']:.2f} | {d['support']} |"
        )
    return "\n".join(lines)


def write_report(metrics: dict, json_path=None, md_path=None,
                 latency: dict | None = None, cost: float | None = None) -> None:
    payload = dict(metrics)
    if latency is not None:
        payload["latency"] = latency
    if cost is not None:
        payload["cost_weighted_error"] = cost
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(payload, indent=2))
    if md_path:
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text(to_markdown(metrics, latency=latency, cost=cost))
