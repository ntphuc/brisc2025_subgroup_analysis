from __future__ import annotations

import argparse
import math
from pathlib import Path
import pandas as pd


def get_row(df: pd.DataFrame, study: str, arch: str, method: str):
    r = df[(df.study == study) & (df.architecture == arch) & (df.method == method)]
    if len(r) != 1:
        raise KeyError(f"Expected one row for {study}/{arch}/{method}, found {len(r)}")
    return r.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare a reproduction run with the paper's archived aggregate targets.")
    ap.add_argument("--aggregate-dir", default="results/paper_suite/aggregate")
    ap.add_argument("--reference-dir", default="reference_results")
    ap.add_argument("--allow-incomplete", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()

    agg_path = Path(args.aggregate_dir) / "multi_seed_summary.csv"
    ref_path = Path(args.reference_dir) / "multi_seed_summary.csv"
    if not agg_path.exists():
        if args.allow_incomplete:
            print(f"SKIP: {agg_path} does not exist yet")
            return
        raise SystemExit(f"Missing aggregate output: {agg_path}")
    cur = pd.read_csv(agg_path)
    ref = pd.read_csv(ref_path)

    checks = [
        ("standard", "unet_effb0", "erm", "dice_mean", 0.010),
        ("standard", "unet_effb0", "erm", "gl_ax_dice_mean", 0.035),
        ("standard", "unet_effb0", "erm", "small_gl_ax_dice_mean", 0.070),
        ("standard", "unetpp_effb0", "erm", "dice_mean", 0.012),
        ("standard", "deeplabv3p_resnet50", "erm", "dice_mean", 0.012),
        ("standard", "unet_effb0", "groupdro", "dice_mean", 0.020),
        ("confirmatory", "unet_effb0", "erm", "dice_mean", 0.012),
        ("confirmatory", "unet_effb0", "balanced", "small_gl_ax_dice_mean", 0.090),
    ]
    rows = []
    failed = 0
    for study, arch, method, metric, tol in checks:
        try:
            a = get_row(cur, study, arch, method)
            b = get_row(ref, study, arch, method)
        except KeyError as exc:
            if args.allow_incomplete:
                rows.append((f"{study}/{arch}/{method}/{metric}", "SKIP", "", "", tol))
                continue
            raise SystemExit(str(exc))
        actual, target = float(a[metric]), float(b[metric])
        diff = abs(actual - target)
        status = "PASS" if diff <= tol else "WARN"
        failed += int(status == "WARN")
        rows.append((f"{study}/{arch}/{method}/{metric}", status, actual, target, tol))

    print("Reproduction comparison (small numeric differences across CUDA/PyTorch versions are expected):")
    for name, status, actual, target, tol in rows:
        if status == "SKIP":
            print(f"  {status:4s} {name}")
        else:
            print(f"  {status:4s} {name}: actual={actual:.6f} paper={target:.6f} tolerance={tol:.3f}")
    if failed:
        print(f"\n{failed} checks are outside the practical tolerance. This is a diagnostic warning, not automatic evidence of an invalid run.")
    else:
        print("\nAll available key metrics are within the practical reproduction tolerances.")


if __name__ == "__main__":
    main()
