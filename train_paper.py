from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import BRISC2025Dataset
from src.model import ARCHITECTURES, build_model
from src.paper_metrics import GROUPS, GROUP_TO_INDEX, hard_batch_metrics, normalize_meta_float, normalize_meta_list, per_sample_bce_dice_loss
from src.utils import get_device, load_yaml, seed_everything


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _loader_kwargs(cfg: dict, device: torch.device) -> dict:
    nw = int(cfg.get("num_workers", 2))
    kw = {"num_workers": nw, "pin_memory": device.type == "cuda"}
    if nw > 0:
        kw["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    return kw


def _resolve_cfg(cfg: dict, architecture: str) -> dict:
    out = dict(cfg)
    override = (cfg.get("architecture_overrides", {}) or {}).get(architecture, {}) or {}
    out.update(override)
    return out


def _manifest_groups(manifest: str, split: str, expected_len: int) -> list[tuple[str, str]] | None:
    try:
        df = pd.read_csv(manifest)
        if "split" not in df.columns or "tumor_code" not in df.columns or "plane_code" not in df.columns:
            return None
        sub = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()
        if len(sub) != expected_len:
            return None
        return list(zip(sub["tumor_code"].astype(str).str.lower(), sub["plane_code"].astype(str).str.lower()))
    except Exception:
        return None


def _scan_dataset_meta(ds: BRISC2025Dataset) -> tuple[list[tuple[str, str]], list[float]]:
    groups: list[tuple[str, str]] = []
    lesion: list[float] = []
    for idx in range(len(ds)):
        _, mask, meta = ds[idx]
        t = str(meta.get("tumor_code", "")).lower()
        p = str(meta.get("plane_code", "")).lower()
        groups.append((t, p))
        # Always use the transformed 256x256 mask. This keeps the train-derived
        # size threshold on the exact same pixel scale as evaluation gt_pixels.
        lesion.append(float(mask.sum().item()))
    return groups, lesion


def _collect_groups_and_sizes(ds: BRISC2025Dataset, manifest: str, split: str) -> tuple[list[tuple[str, str]], list[float]]:
    manifest_groups = _manifest_groups(manifest, split, len(ds))
    scanned_groups, scanned_lesion = _scan_dataset_meta(ds)
    return (manifest_groups if manifest_groups is not None else scanned_groups), scanned_lesion


def _balanced_sampler(groups: list[tuple[str, str]], exponent: float, seed: int) -> WeightedRandomSampler:
    counts = Counter(groups)
    weights = [1.0 / (max(1, counts[g]) ** float(exponent)) for g in groups]
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True, generator=gen)


def _groupdro_loss(per_sample: torch.Tensor, tumors: list[str], planes: list[str], q: torch.Tensor, eta: float, min_q: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    present: list[int] = []
    group_losses: dict[int, torch.Tensor] = {}
    for g, idx in GROUP_TO_INDEX.items():
        mask = torch.tensor(
            [t == g[0] and p == g[1] for t, p in zip(tumors, planes)],
            device=per_sample.device,
            dtype=torch.bool,
        )
        if mask.any():
            present.append(idx)
            group_losses[idx] = per_sample[mask].mean()
    if not present:
        return per_sample.mean(), q, {}

    with torch.no_grad():
        for idx in present:
            q[idx] = q[idx] * torch.exp(float(eta) * group_losses[idx].detach())
        q.clamp_(min=float(min_q))
        q /= q.sum()

    q_present = q[present]
    q_present = q_present / q_present.sum()
    robust = torch.zeros((), device=per_sample.device)
    detail = {}
    for local_i, idx in enumerate(present):
        robust = robust + q_present[local_i] * group_losses[idx]
        detail[f"{GROUPS[idx][0]}-{GROUPS[idx][1]}"] = float(group_losses[idx].detach().cpu())
    return robust, q, detail


def _batch_group_meta(meta: dict, n: int) -> tuple[list[str], list[str]]:
    return normalize_meta_list(meta["tumor_code"], n), normalize_meta_list(meta["plane_code"], n)


def _epoch_train(model, loader, optimizer, scaler, device, cfg, method: str, q: torch.Tensor | None, epoch: int):
    model.train()
    losses, dices, ious = [], [], []
    eta = float((cfg.get("groupdro", {}) or {}).get("eta", 0.05))
    min_q = float((cfg.get("groupdro", {}) or {}).get("min_q", 1e-6))
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    for images, masks, meta in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            per_sample = per_sample_bce_dice_loss(
                logits, masks,
                float(cfg.get("bce_weight", 0.5)),
                float(cfg.get("dice_weight", 0.5)),
            )
            if method == "groupdro":
                tumors, planes = _batch_group_meta(meta, len(images))
                loss, q, _ = _groupdro_loss(per_sample, tumors, planes, q, eta, min_q)
            else:
                loss = per_sample.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite train loss at epoch {epoch}: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        m = hard_batch_metrics(logits.detach(), masks, float(cfg.get("threshold", 0.5)))
        losses.append(float(loss.detach().cpu()))
        dices.extend(m["dice"].detach().cpu().tolist())
        ious.extend(m["iou"].detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
    }, q


@torch.no_grad()
def _epoch_eval(model, loader, device, cfg):
    model.eval()
    losses, dices, ious, recalls = [], [], [], []
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            per_sample = per_sample_bce_dice_loss(
                logits, masks,
                float(cfg.get("bce_weight", 0.5)),
                float(cfg.get("dice_weight", 0.5)),
            )
        m = hard_batch_metrics(logits, masks, float(cfg.get("threshold", 0.5)))
        losses.extend(per_sample.detach().cpu().tolist())
        dices.extend(m["dice"].detach().cpu().tolist())
        ious.extend(m["iou"].detach().cpu().tolist())
        recalls.extend(m["recall"].detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "recall": float(np.mean(recalls)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-strength BRISC2025 training: ERM, balanced sampling, GroupDRO.")
    ap.add_argument("--config", default="configs/paper_suite.yaml")
    ap.add_argument("--manifest", default="data/brisc2025_manifest.csv")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--architecture", choices=sorted(ARCHITECTURES), required=True)
    ap.add_argument("--method", choices=["erm", "balanced", "groupdro"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = _resolve_cfg(load_yaml(args.config), args.architecture)
    cfg["architecture"] = args.architecture
    cfg["method"] = args.method
    cfg["seed"] = int(args.seed)
    cfg["train_split"] = args.train_split
    cfg["val_split"] = args.val_split
    seed_everything(args.seed)
    device = get_device(args.device)

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_ds = BRISC2025Dataset(
        args.manifest, args.train_split, args.data_root, int(cfg["image_size"]), True,
        bool(cfg.get("imagenet_norm", True)),
    )
    # A non-augmented mirror is used only for immutable group labels and lesion
    # sizes. This prevents random training augmentation from changing the q33/q67
    # size thresholds across model seeds.
    train_meta_ds = BRISC2025Dataset(
        args.manifest, args.train_split, args.data_root, int(cfg["image_size"]), False,
        bool(cfg.get("imagenet_norm", True)),
    )
    val_ds = BRISC2025Dataset(
        args.manifest, args.val_split, args.data_root, int(cfg["image_size"]), False,
        bool(cfg.get("imagenet_norm", True)),
    )
    groups, lesion_sizes = _collect_groups_and_sizes(train_meta_ds, args.manifest, args.train_split)
    if len(groups) != len(train_ds):
        raise RuntimeError("Group metadata length does not match train dataset")
    lesion_arr = np.asarray(lesion_sizes, dtype=float)
    small_q = float(cfg.get("small_quantile", 0.33))
    small_threshold = float(np.quantile(lesion_arr, small_q))
    large_threshold = float(np.quantile(lesion_arr, 1.0 - small_q))
    counts = Counter(groups)

    kw = _loader_kwargs(cfg, device)
    gen = torch.Generator().manual_seed(args.seed)
    if args.method == "balanced":
        exponent = float((cfg.get("balanced", {}) or {}).get("inverse_frequency_exponent", 1.0))
        sampler = _balanced_sampler(groups, exponent, args.seed)
        train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), sampler=sampler, drop_last=False, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, generator=gen, drop_last=False, **kw)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=False, **kw)

    model = build_model(args.architecture, pretrained_encoder=bool(cfg.get("pretrained_encoder", True))).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["epochs"])), eta_min=float(cfg.get("min_lr", 0.0)))
    scaler = GradScaler("cuda", enabled=bool(cfg.get("amp", True) and device.type == "cuda"))
    q = torch.ones(len(GROUPS), dtype=torch.float32, device=device) / len(GROUPS) if args.method == "groupdro" else None

    print(f"TRAIN paper-suite | arch={args.architecture} method={args.method} seed={args.seed} device={device}", flush=True)
    print(f"samples train={len(train_ds)} val={len(val_ds)} params={n_params:,} small_threshold={small_threshold:.2f}", flush=True)
    print("group_counts=" + json.dumps({f"{a}-{b}": int(counts[(a, b)]) for a, b in GROUPS}), flush=True)

    history_path = run_dir / "history.csv"
    fields = ["epoch", "lr", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou", "val_recall", "seconds"]
    best = -float("inf")
    bad = 0
    best_epoch = 0
    patience = int(cfg.get("patience", 8))
    min_delta = float(cfg.get("min_delta", 0.0005))

    with history_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for epoch in range(1, int(cfg["epochs"]) + 1):
            t0 = time.time()
            tr, q = _epoch_train(model, train_loader, optimizer, scaler, device, cfg, args.method, q, epoch)
            va = _epoch_eval(model, val_loader, device, cfg)
            sec = time.time() - t0
            lr = float(optimizer.param_groups[0]["lr"])
            wr.writerow({"epoch": epoch, "lr": lr, "train_loss": tr["loss"], "train_dice": tr["dice"], "train_iou": tr["iou"], **{f"val_{k}": v for k, v in va.items()}, "seconds": sec})
            f.flush()
            print(
                f"EPOCH {epoch:02d}/{int(cfg['epochs'])} | train={tr['loss']:.4f} dice={tr['dice']:.4f} "
                f"| val={va['loss']:.4f} dice={va['dice']:.4f} recall={va['recall']:.4f} "
                f"| lr={lr:.2e} time={sec:.1f}s",
                flush=True,
            )
            if va["dice"] > best + min_delta:
                best = float(va["dice"])
                best_epoch = epoch
                bad = 0
                payload = {
                    "model": model.state_dict(),
                    "config": cfg,
                    "architecture": args.architecture,
                    "method": args.method,
                    "seed": args.seed,
                    "epoch": epoch,
                    "best_val_dice": best,
                    "manifest": str(args.manifest),
                    "train_split": args.train_split,
                    "val_split": args.val_split,
                    "training_small_threshold_gt_pixels": small_threshold,
                    "training_large_threshold_gt_pixels": large_threshold,
                    "group_counts": {f"{a}-{b}": int(counts[(a, b)]) for a, b in GROUPS},
                    "groupdro_q": q.detach().cpu().tolist() if q is not None else None,
                }
                torch.save(payload, run_dir / "best.pt")
            else:
                bad += 1
                if bad >= patience:
                    print(f"EARLY STOP epoch={epoch} best_epoch={best_epoch} best_val_dice={best:.6f}", flush=True)
                    break
            scheduler.step()

    summary = {
        "architecture": args.architecture,
        "method": args.method,
        "seed": int(args.seed),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "parameters": int(n_params),
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best),
        "training_small_threshold_gt_pixels": small_threshold,
        "training_large_threshold_gt_pixels": large_threshold,
        "group_counts": {f"{a}-{b}": int(counts[(a, b)]) for a, b in GROUPS},
        "groupdro_q": q.detach().cpu().tolist() if q is not None else None,
        "training_complete": True,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TRAINING COMPLETE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
