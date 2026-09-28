from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TUMORS = ["gl", "me", "pi"]
PLANES = ["ax", "co", "sa"]
TUMOR_LABELS = {"gl": "Glioma", "me": "Meningioma", "pi": "Pituitary"}
PLANE_LABELS = {"ax": "Axial", "co": "Coronal", "sa": "Sagittal"}
ARCH_LABELS = {
    "unet_effb0": "U-Net / EfficientNet-B0",
    "unetpp_effb0": "U-Net++ / EfficientNet-B0",
    "deeplabv3p_resnet50": "DeepLabV3+ / ResNet50",
}


def save_table2(ms: pd.DataFrame, out: Path) -> None:
    s = ms[ms.study == "standard"].copy()
    keep = []
    for arch, method in [
        ("deeplabv3p_resnet50", "erm"),
        ("unet_effb0", "erm"),
        ("unetpp_effb0", "erm"),
        ("unet_effb0", "balanced"),
        ("unet_effb0", "groupdro"),
    ]:
        r = s[(s.architecture == arch) & (s.method == method)]
        if len(r):
            keep.append(r.iloc[0])
    rows = []
    for r in keep:
        rows.append({
            "Architecture": ARCH_LABELS.get(r.architecture, r.architecture),
            "Method": r.method,
            "Dice_mean": r.dice_mean,
            "Dice_std": r.dice_std,
            "Worst_mean": r.worst_group_dice_mean,
            "Worst_std": r.worst_group_dice_std,
            "GL_AX_mean": r.gl_ax_dice_mean,
            "GL_AX_std": r.gl_ax_dice_std,
            "Small_GL_AX_mean": r.small_gl_ax_dice_mean,
            "Small_GL_AX_std": r.small_gl_ax_dice_std,
            "P10_mean": r.p10_dice_mean,
            "P10_std": r.p10_dice_std,
            "HD95_mean": r.hd95_mean,
            "HD95_std": r.hd95_std,
        })
    pd.DataFrame(rows).to_csv(out / "table2_standard_performance.csv", index=False)


def save_table3(effects: pd.DataFrame, out: Path) -> None:
    label = {"tumor": "Tumor type", "size": "Lesion size (log area)", "plane": "Imaging plane", "interaction": "Tumor x plane"}
    rows = []
    for factor in ("tumor", "size", "plane", "interaction"):
        row = {"Factor": label[factor]}
        for study in ("standard", "confirmatory"):
            r = effects[(effects.study == study) & (effects.factor == factor)]
            if len(r):
                x = r.iloc[0]
                row[f"{study}_eta2"] = x.partial_eta2_mean
                row[f"{study}_classic_sig"] = f"{int(x.classic_significant_runs)}/{int(x.runs)}"
                row[f"{study}_hc3_sig"] = f"{int(x.hc3_significant_runs)}/{int(x.runs)}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "table3_effect_recurrence.csv", index=False)


def save_table4(paired: pd.DataFrame, out: Path) -> None:
    rows = []
    for study in ("standard", "confirmatory"):
        for method in ("balanced", "groupdro"):
            row = {"Study": study, "Method": method}
            for subset, prefix in (("all", "Overall"), ("gl_ax", "GL_AX"), ("small_gl_ax", "Small_GL_AX")):
                r = paired[(paired.study == study) & (paired.method == method) & (paired.subset == subset)]
                if len(r):
                    x = r.iloc[0]
                    row[f"{prefix}_delta"] = x.mean_delta_dice
                    row[f"{prefix}_ci_low"] = x.ci95_low
                    row[f"{prefix}_ci_high"] = x.ci95_high
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "table4_paired_deltas.csv", index=False)


def figure1(groups: pd.DataFrame, out: Path) -> None:
    d = groups[(groups.study == "standard") & (groups.method == "erm")]
    archs = ["unet_effb0", "unetpp_effb0", "deeplabv3p_resnet50"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    values = []
    for arch in archs:
        a = d[d.architecture == arch].groupby(["tumor_code", "plane_code"], as_index=False).agg(dice=("dice", "mean"), n=("n", "max"))
        values.extend(a.dice.tolist())
    vmin, vmax = min(values), max(values)
    for ax, arch in zip(axes, archs):
        a = d[d.architecture == arch].groupby(["tumor_code", "plane_code"], as_index=False).agg(dice=("dice", "mean"), n=("n", "max"))
        mat = np.full((3, 3), np.nan)
        nmat = np.zeros((3, 3), dtype=int)
        for _, r in a.iterrows():
            i, j = TUMORS.index(r.tumor_code), PLANES.index(r.plane_code)
            mat[i, j] = r.dice
            nmat[i, j] = int(r.n)
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(ARCH_LABELS[arch])
        ax.set_xticks(range(3), [PLANE_LABELS[p] for p in PLANES])
        ax.set_yticks(range(3), [TUMOR_LABELS[t] for t in TUMORS])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.3f}\n(n={nmat[i,j]})", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.86, label="Mean Dice")
    fig.savefig(out / "fig1_group_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure2(effects: pd.DataFrame, out: Path) -> None:
    order = ["tumor", "size", "plane", "interaction"]
    labels = ["Tumor type", "Lesion size", "Imaging plane", "Tumor x plane"]
    std = [float(effects[(effects.study == "standard") & (effects.factor == f)].partial_eta2_mean.iloc[0]) for f in order]
    conf = [float(effects[(effects.study == "confirmatory") & (effects.factor == f)].partial_eta2_mean.iloc[0]) for f in order]
    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.3, 4.1))
    ax.bar(x - w/2, std, width=w, label="Standard exploratory")
    ax.bar(x + w/2, conf, width=w, label="Locked confirmatory")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Mean partial eta squared")
    ax.legend()
    ax.set_title("Size-controlled effect magnitude across protocols")
    fig.tight_layout()
    fig.savefig(out / "fig2_effect_sizes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure3(ms: pd.DataFrame, out: Path) -> None:
    d = ms[(ms.architecture == "unet_effb0") & (ms.method.isin(["erm", "balanced", "groupdro"]))]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for _, r in d.iterrows():
        ax.errorbar(
            r.dice_mean, r.small_gl_ax_dice_mean,
            xerr=r.dice_std, yerr=r.small_gl_ax_dice_std,
            fmt="o", capsize=3,
            label=f"{r.study}-{r.method}",
        )
        ax.annotate(f"{r.study[:3]}-{r.method}", (r.dice_mean, r.small_gl_ax_dice_mean), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Overall Dice")
    ax.set_ylabel("Small glioma-axial Dice")
    ax.set_title("Mitigation trade-off: subgroup gains are not free")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig3_mitigation_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure4(q: pd.DataFrame, out: Path) -> None:
    if len(q) == 0:
        return
    q = q.copy()
    q["group"] = q.tumor_code.str.upper() + "-" + q.plane_code.str.upper()
    order = [f"{t.upper()}-{p.upper()}" for t in TUMORS for p in PLANES]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True, constrained_layout=True)
    for ax, study in zip(axes, ("standard", "confirmatory")):
        s = q[q.study == study]
        if len(s) == 0:
            continue
        piv = s.pivot(index="group", columns="seed", values="weight").reindex(order)
        x = np.arange(len(order))
        width = 0.24
        for k, seed in enumerate(sorted(piv.columns)):
            ax.bar(x + (k-1)*width, piv[seed].values, width=width, label=f"seed {seed}")
        ax.set_xticks(x, order, rotation=55, ha="right")
        ax.set_title(study.capitalize())
        ax.set_ylabel("Final adversarial weight")
        ax.legend(fontsize=8)
    fig.savefig(out / "fig4_groupdro_weights.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recreate the data tables and figures used by the paper from aggregate outputs.")
    ap.add_argument("--aggregate-dir", default="results/paper_suite/aggregate")
    ap.add_argument("--output-dir", default="results/paper_suite/paper_artifacts")
    args = ap.parse_args()
    root = Path(args.aggregate_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ms = pd.read_csv(root / "multi_seed_summary.csv")
    groups = pd.read_csv(root / "per_run_group_summary.csv")
    effects = pd.read_csv(root / "effect_recurrence.csv")
    paired = pd.read_csv(root / "paired_method_tests.csv")
    q = pd.read_csv(root / "groupdro_final_weights.csv") if (root / "groupdro_final_weights.csv").exists() else pd.DataFrame()

    save_table2(ms, out)
    save_table3(effects, out)
    save_table4(paired, out)
    figure1(groups, out)
    figure2(effects, out)
    figure3(ms, out)
    figure4(q, out)
    print(f"Paper tables/figures written to {out}")


if __name__ == "__main__":
    main()
