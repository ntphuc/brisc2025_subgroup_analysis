from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

GROUPS = [(t, p) for t in ("gl", "me", "pi") for p in ("ax", "co", "sa")]
GROUP_TO_INDEX = {g: i for i, g in enumerate(GROUPS)}


def normalize_meta_list(value, n: int) -> list[str]:
    if torch.is_tensor(value):
        vals = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        vals = list(value)
    else:
        vals = [value] * n
    if len(vals) != n:
        raise ValueError(f"metadata length mismatch: expected {n}, got {len(vals)}")
    return [str(x).lower() for x in vals]


def normalize_meta_float(value, n: int) -> list[float]:
    if torch.is_tensor(value):
        vals = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        vals = list(value)
    else:
        vals = [value] * n
    if len(vals) != n:
        raise ValueError(f"metadata length mismatch: expected {n}, got {len(vals)}")
    return [float(x) for x in vals]


def per_sample_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = bce.flatten(1).mean(1)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).flatten(1).sum(1)
    den = probs.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice_loss = 1.0 - (2.0 * inter + eps) / (den + eps)
    return float(bce_weight) * bce + float(dice_weight) * dice_loss


def hard_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    pred = (torch.sigmoid(logits) >= float(threshold)).float()
    dims = (1, 2, 3)
    tp = (pred * targets).sum(dims)
    fp = (pred * (1.0 - targets)).sum(dims)
    fn = ((1.0 - pred) * targets).sum(dims)
    tn = ((1.0 - pred) * (1.0 - targets)).sum(dims)
    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "specificity": specificity, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_pixels": pred.sum(dims), "gt_pixels": targets.sum(dims), "pred": pred,
    }


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    h, w = pred.shape[-2:]
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        penalty = float(math.hypot(h, w))
        return penalty, penalty
    pred_surface = pred ^ binary_erosion(pred, border_value=0)
    gt_surface = gt ^ binary_erosion(gt, border_value=0)
    if not pred_surface.any():
        pred_surface = pred
    if not gt_surface.any():
        gt_surface = gt
    dt_to_gt = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)
    d_pg = dt_to_gt[pred_surface]
    d_gp = dt_to_pred[gt_surface]
    distances = np.concatenate([d_pg, d_gp]).astype(float)
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def reliability_summary(
    rows: Iterable[dict], small_threshold: float | None = None, min_group_count: int = 3
) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    dice = np.asarray([float(r["dice"]) for r in rows], dtype=float)
    groups: dict[tuple[str, str], list[float]] = {}
    small_groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        key = (str(r["tumor_code"]).lower(), str(r["plane_code"]).lower())
        groups.setdefault(key, []).append(float(r["dice"]))
        if small_threshold is not None and float(r.get("gt_pixels", np.inf)) <= float(small_threshold):
            small_groups.setdefault(key, []).append(float(r["dice"]))
    means = {k: float(np.mean(v)) for k, v in groups.items() if len(v) >= min_group_count}
    smeans = {k: float(np.mean(v)) for k, v in small_groups.items() if len(v) >= min_group_count}
    glax = means.get(("gl", "ax"), float("nan"))
    sglax = smeans.get(("gl", "ax"), float("nan"))
    return {
        "mean_dice": float(np.mean(dice)),
        "std_dice": float(np.std(dice, ddof=1)) if len(dice) > 1 else 0.0,
        "median_dice": float(np.median(dice)),
        "p10_dice": float(np.quantile(dice, 0.10)),
        "bottom20_dice": float(np.mean(np.sort(dice)[: max(1, int(math.ceil(0.2 * len(dice))))])),
        "failure_lt_0p8": float(np.mean(dice < 0.8)),
        "failure_lt_0p5": float(np.mean(dice < 0.5)),
        "failure_lt_0p1": float(np.mean(dice < 0.1)),
        "worst_group_dice": float(min(means.values())) if means else float("nan"),
        "best_group_dice": float(max(means.values())) if means else float("nan"),
        "group_gap": float(max(means.values()) - min(means.values())) if means else float("nan"),
        "gl_ax_dice": glax,
        "small_gl_ax_dice": sglax,
        "small_worst_group_dice": float(min(smeans.values())) if smeans else float("nan"),
    }
