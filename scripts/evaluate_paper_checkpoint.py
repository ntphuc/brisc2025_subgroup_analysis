from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.dataset import BRISC2025Dataset
from src.model import build_model
from src.paper_metrics import hard_batch_metrics, normalize_meta_list, reliability_summary, surface_distances
from src.utils import get_device


def _meta_at(meta: dict, key: str, i: int, n: int, default=""):
    if key not in meta:
        return default
    v = meta[key]
    if torch.is_tensor(v):
        vals = v.detach().cpu().reshape(-1).tolist()
        return vals[i] if len(vals) == n else vals[0]
    if isinstance(v, (list, tuple)):
        return v[i] if len(v) == n else v[0]
    return v


def _make_loader(manifest: str, split: str, data_root: str, cfg: dict, device: torch.device):
    ds = BRISC2025Dataset(
        manifest, split, data_root, int(cfg.get("image_size", 256)), False,
        bool(cfg.get("imagenet_norm", True)),
    )
    nw = int(cfg.get("num_workers", 2))
    kw = {"num_workers": nw, "pin_memory": device.type == "cuda"}
    if nw > 0:
        kw["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    return ds, DataLoader(ds, batch_size=int(cfg.get("batch_size", 8)), shuffle=False, **kw)


@torch.no_grad()
def calibrate(model, loader, device, cfg, thresholds: np.ndarray):
    sums = np.zeros(len(thresholds), dtype=np.float64)
    n_total = 0
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            probs = torch.sigmoid(model(images))
        for j, thr in enumerate(thresholds):
            pred = (probs >= float(thr)).float()
            dims = (1, 2, 3)
            tp = (pred * masks).sum(dims)
            fp = (pred * (1.0 - masks)).sum(dims)
            fn = ((1.0 - pred) * masks).sum(dims)
            dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
            sums[j] += float(dice.sum().detach().cpu())
        n_total += int(images.shape[0])
    means = sums / max(1, n_total)
    idx = int(np.argmax(means))
    return float(thresholds[idx]), pd.DataFrame({"threshold": thresholds, "mean_dice": means})


@torch.no_grad()
def evaluate(model, loader, device, cfg, threshold: float):
    rows = []
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    for images, masks, meta in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        m = hard_batch_metrics(logits, masks, threshold)
        pred = m.pop("pred")
        n = int(images.shape[0])
        tumors = normalize_meta_list(meta.get("tumor_code", ""), n)
        planes = normalize_meta_list(meta.get("plane_code", ""), n)
        for i in range(n):
            p_np = pred[i, 0].detach().cpu().numpy().astype(bool)
            g_np = masks[i, 0].detach().cpu().numpy().astype(bool)
            hd95, assd = surface_distances(p_np, g_np)
            t = tumors[i]
            p = planes[i]
            plane_label = {"ax": "axial", "co": "coronal", "sa": "sagittal"}.get(p, p)
            tumor_label = {"gl": "glioma", "me": "meningioma", "pi": "pituitary"}.get(t, t)
            row = {
                "case_id": str(_meta_at(meta, "case_id", i, n, f"case_{len(rows):05d}")),
                "plane_code": p,
                "plane_label": plane_label,
                "tumor_code": t,
                "tumor_label": tumor_label,
                "image_path": str(_meta_at(meta, "image_path", i, n, "")),
                "mask_path": str(_meta_at(meta, "mask_path", i, n, "")),
                "dice": float(m["dice"][i].detach().cpu()),
                "iou": float(m["iou"][i].detach().cpu()),
                "precision": float(m["precision"][i].detach().cpu()),
                "recall": float(m["recall"][i].detach().cpu()),
                "specificity": float(m["specificity"][i].detach().cpu()),
                "hd95": hd95,
                "assd": assd,
                "gt_pixels": int(round(float(m["gt_pixels"][i].detach().cpu()))),
                "pred_pixels": int(round(float(m["pred_pixels"][i].detach().cpu()))),
                "tp": int(round(float(m["tp"][i].detach().cpu()))),
                "fp": int(round(float(m["fp"][i].detach().cpu()))),
                "fn": int(round(float(m["fn"][i].detach().cpu()))),
                "tn": int(round(float(m["tn"][i].detach().cpu()))),
            }
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--calibrate-split", default="val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    architecture = ckpt.get("architecture") or cfg.get("architecture") or "unet_effb0"
    device = get_device(args.device)
    model = build_model(architecture, pretrained_encoder=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    threshold = float(args.threshold if args.threshold is not None else cfg.get("threshold", 0.5))

    if args.calibrate:
        _, val_loader = _make_loader(args.manifest, args.calibrate_split, args.data_root, cfg, device)
        tcfg = cfg.get("threshold_calibration", {}) or {}
        start = float(tcfg.get("start", 0.30))
        stop = float(tcfg.get("stop", 0.70))
        step = float(tcfg.get("step", 0.025))
        thresholds = np.arange(start, stop + 0.5 * step, step)
        threshold, grid = calibrate(model, val_loader, device, cfg, thresholds)
        grid.to_csv(out / "threshold_grid.csv", index=False)
        (out / "selected_threshold.json").write_text(
            json.dumps({"threshold": threshold, "calibration_split": args.calibrate_split, "objective": "mean_dice"}, indent=2),
            encoding="utf-8",
        )

    ds, loader = _make_loader(args.manifest, args.eval_split, args.data_root, cfg, device)
    rows = evaluate(model, loader, device, cfg, threshold)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_case_metrics.csv", index=False)

    small_threshold = ckpt.get("training_small_threshold_gt_pixels")
    rel = reliability_summary(rows, small_threshold=float(small_threshold) if small_threshold is not None else None)
    summary = {
        "architecture": architecture,
        "method": ckpt.get("method", cfg.get("method", "unknown")),
        "seed": int(ckpt.get("seed", cfg.get("seed", -1))),
        "split": args.eval_split,
        "n": int(len(df)),
        "threshold": float(threshold),
        "dice": float(df["dice"].mean()),
        "iou": float(df["iou"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "specificity": float(df["specificity"].mean()),
        "hd95": float(df["hd95"].mean()),
        "assd": float(df["assd"].mean()),
        "training_small_threshold_gt_pixels": float(small_threshold) if small_threshold is not None else None,
        **rel,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
