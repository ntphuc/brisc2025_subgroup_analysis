from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.statistics import ancova_with_hc3, prepare_ancova_frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Size-controlled tumor/plane analysis used in the paper.")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--training-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics)
    ts = json.loads(Path(args.training_summary).read_text(encoding="utf-8"))
    q33 = float(ts["training_small_threshold_gt_pixels"])
    q67 = float(ts["training_large_threshold_gt_pixels"])

    def size_regime(x: float) -> str:
        if float(x) <= q33:
            return "small"
        if float(x) <= q67:
            return "medium"
        return "large"

    d = prepare_ancova_frame(df)
    d["size_regime"] = d["gt_pixels"].map(size_regime)
    d.to_csv(out / "per_case_with_training_size_regime.csv", index=False)

    grouped = (
        d.groupby(["tumor_code", "plane_code", "size_regime"])["dice"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.to_csv(out / "by_tumor_plane_size.csv", index=False)

    for regime in ("small", "medium", "large"):
        sub = grouped[grouped["size_regime"] == regime]
        mat = (
            sub.pivot(index="tumor_code", columns="plane_code", values="mean")
            .reindex(index=["gl", "me", "pi"], columns=["ax", "co", "sa"])
        )
        mat.to_csv(out / f"dice_matrix_{regime}_3x3.csv")

    small = grouped[grouped.size_regime == "small"].copy()
    small.to_csv(out / "small_lesion_tumor_plane.csv", index=False)

    classic, hc3, factors = ancova_with_hc3(d)
    classic.to_csv(out / "ancova_dice_tumor_plane_logsize.csv")
    hc3.to_csv(out / "ancova_dice_tumor_plane_logsize_hc3.csv")

    factor_rows = []
    for factor, values in factors.items():
        factor_rows.append({"factor": factor, **values})
    pd.DataFrame(factor_rows).to_csv(out / "ancova_factor_summary.csv", index=False)

    glax_small = small[(small.tumor_code == "gl") & (small.plane_code == "ax")]
    glax_small_dice = float(glax_small["mean"].iloc[0]) if len(glax_small) else float("nan")
    inter = factors["interaction"]
    summary = {
        "small_threshold_gt_pixels_train_q33": q33,
        "large_threshold_gt_pixels_train_q67": q67,
        "threshold_source": "training masks only",
        "ancova_formula": "dice ~ C(tumor_code) * C(plane_code) + log1p(gt_pixels)",
        "ancova_interaction_F": inter["F_classic"],
        "ancova_interaction_p": inter["p_classic"],
        "ancova_interaction_partial_eta2": inter["partial_eta2"],
        "ancova_interaction_F_hc3": inter["F_hc3"],
        "ancova_interaction_p_hc3": inter["p_hc3"],
        "factors": factors,
        "small_gl_ax_dice": glax_small_dice,
        "n": int(len(d)),
    }
    (out / "size_controlled_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
