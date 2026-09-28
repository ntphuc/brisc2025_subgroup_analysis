from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.statistics import ancova_with_hc3


def bootstrap_ci(x: np.ndarray, seed: int = 2026, n_boot: int = 10000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cohen_dz(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    s = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
    return float(np.mean(delta) / s) if s > 0 else 0.0


def parse_result_path(path: Path, root: Path):
    rel = path.relative_to(root)
    # expected: standard/<arch>/<method>/seed_42/calibrated/per_case_metrics.csv
    parts = rel.parts
    if len(parts) < 6:
        return None
    study, arch, method, seed_part, variant = parts[:5]
    m = re.match(r"seed_(\d+)", seed_part)
    if not m:
        return None
    return study, arch, method, int(m.group(1)), variant


def training_summary(outputs_root: Path, study: str, arch: str, method: str, seed: int) -> dict:
    p = outputs_root / study / arch / method / f"seed_{seed}" / "training_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_run(df: pd.DataFrame, small_thr: float | None) -> tuple[dict, list[dict]]:
    group = df.groupby(["tumor_code", "plane_code"])["dice"].agg(["mean", "count"]).reset_index()
    valid = group[group["count"] >= 3]
    glax = valid[(valid.tumor_code == "gl") & (valid.plane_code == "ax")]
    glax_dice = float(glax["mean"].iloc[0]) if len(glax) else float("nan")
    worst = float(valid["mean"].min()) if len(valid) else float("nan")
    best = float(valid["mean"].max()) if len(valid) else float("nan")
    if len(glax):
        ranks = valid["mean"].rank(method="min", ascending=True)
        glax_rank = float(ranks[(valid.tumor_code == "gl") & (valid.plane_code == "ax")].iloc[0])
    else:
        glax_rank = float("nan")

    small_glax = float("nan")
    small_worst = float("nan")
    if small_thr is not None and math.isfinite(float(small_thr)):
        s = df[df["gt_pixels"] <= float(small_thr)].copy()
        sg = s.groupby(["tumor_code", "plane_code"])["dice"].agg(["mean", "count"]).reset_index()
        sg = sg[sg["count"] >= 3]
        r = sg[(sg.tumor_code == "gl") & (sg.plane_code == "ax")]
        small_glax = float(r["mean"].iloc[0]) if len(r) else float("nan")
        small_worst = float(sg["mean"].min()) if len(sg) else float("nan")

    dice = df["dice"].to_numpy(float)
    k20 = max(1, int(math.ceil(0.2 * len(dice))))
    _, _, factors = ancova_with_hc3(df)
    factor_rows = [{"factor": k, **v} for k, v in factors.items()]
    out = {
        "n": int(len(df)),
        "dice": float(np.mean(dice)),
        "iou": float(df["iou"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "specificity": float(df["specificity"].mean()) if "specificity" in df else float("nan"),
        "hd95": float(df["hd95"].mean()),
        "assd": float(df["assd"].mean()),
        "p10_dice": float(np.quantile(dice, 0.10)),
        "bottom20_dice": float(np.mean(np.sort(dice)[:k20])),
        "failure_lt_0p8": float(np.mean(dice < 0.8)),
        "failure_lt_0p5": float(np.mean(dice < 0.5)),
        "failure_lt_0p1": float(np.mean(dice < 0.1)),
        "worst_group_dice": worst,
        "best_group_dice": best,
        "group_gap": best - worst if math.isfinite(best) and math.isfinite(worst) else float("nan"),
        "gl_ax_dice": glax_dice,
        "gl_ax_rank_from_worst": glax_rank,
        "small_gl_ax_dice": small_glax,
        "small_worst_group_dice": small_worst,
    }
    for factor, v in factors.items():
        out[f"{factor}_partial_eta2"] = v["partial_eta2"]
        out[f"{factor}_p_classic"] = v["p_classic"]
        out[f"{factor}_p_hc3"] = v["p_hc3"]
    # Backward-compatible names used by earlier analysis scripts.
    out["interaction_p"] = out["interaction_p_classic"]
    out["interaction_p_hc3"] = out["interaction_p_hc3"]
    out["interaction_partial_eta2"] = out["interaction_partial_eta2"]
    return out, factor_rows


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    if not math.isfinite(float(mean)):
        return "NA"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def avg_method_over_seeds(raw: dict, study: str, arch: str, method: str) -> pd.DataFrame | None:
    dfs = []
    for (st, ar, me, seed), df in raw.items():
        if st != study or ar != arch or me != method:
            continue
        cols = ["case_id", "tumor_code", "plane_code", "gt_pixels", "dice"]
        part = df[cols].copy().rename(columns={"dice": f"dice_{seed}"})
        dfs.append(part)
    if not dfs:
        return None
    m = dfs[0]
    for d in dfs[1:]:
        m = m.merge(d[["case_id", [c for c in d.columns if c.startswith("dice_")][0]]], on="case_id", how="inner")
    dice_cols = [c for c in m.columns if c.startswith("dice_")]
    m["dice"] = m[dice_cols].mean(axis=1)
    return m[["case_id", "tumor_code", "plane_code", "gt_pixels", "dice"]]


def paired_tests(raw: dict, run_df: pd.DataFrame, primary_arch: str) -> pd.DataFrame:
    rows: list[dict] = []
    for study in sorted({k[0] for k in raw}):
        methods = sorted({k[2] for k in raw if k[0] == study and k[1] == primary_arch})
        if "erm" not in methods:
            continue
        base = avg_method_over_seeds(raw, study, primary_arch, "erm")
        if base is None:
            continue
        thr_rows = run_df[
            (run_df.study == study)
            & (run_df.architecture == primary_arch)
            & (run_df.method == "erm")
        ]["small_threshold"].dropna()
        small_thr = float(thr_rows.iloc[0]) if len(thr_rows) else float("nan")
        for method in methods:
            if method == "erm":
                continue
            comp = avg_method_over_seeds(raw, study, primary_arch, method)
            if comp is None:
                continue
            paired = base.rename(columns={"dice": "dice_erm"}).merge(
                comp[["case_id", "dice"]].rename(columns={"dice": "dice_method"}), on="case_id", how="inner"
            )
            masks = {
                "all": np.ones(len(paired), dtype=bool),
                "gl_ax": (paired.tumor_code == "gl") & (paired.plane_code == "ax"),
                "small_gl_ax": (paired.tumor_code == "gl") & (paired.plane_code == "ax") & (paired.gt_pixels <= small_thr),
            }
            for subset, mask in masks.items():
                d = paired.loc[mask, "dice_method"].to_numpy(float) - paired.loc[mask, "dice_erm"].to_numpy(float)
                if len(d) == 0:
                    continue
                try:
                    pval = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
                except Exception:
                    pval = float("nan")
                lo, hi = bootstrap_ci(d)
                rows.append(
                    {
                        "study": study,
                        "architecture": primary_arch,
                        "method": method,
                        "vs": "erm",
                        "subset": subset,
                        "n": int(len(d)),
                        "mean_delta_dice": float(np.mean(d)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "wilcoxon_p": pval,
                        "cohen_dz": cohen_dz(d),
                    }
                )
    return pd.DataFrame(rows)


def effect_recurrence(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "standard": run_df[(run_df.study == "standard") & (run_df.method == "erm")],
        "confirmatory": run_df[(run_df.study == "confirmatory") & (run_df.method == "erm") & (run_df.architecture == "unet_effb0")],
    }
    for study, sub in scopes.items():
        if len(sub) == 0:
            continue
        for factor in ("tumor", "size", "plane", "interaction"):
            rows.append(
                {
                    "study": study,
                    "factor": factor,
                    "runs": int(len(sub)),
                    "partial_eta2_mean": float(sub[f"{factor}_partial_eta2"].mean()),
                    "classic_significant_runs": int((sub[f"{factor}_p_classic"] < 0.05).sum()),
                    "hc3_significant_runs": int((sub[f"{factor}_p_hc3"] < 0.05).sum()),
                }
            )
    return pd.DataFrame(rows)


def groupdro_weights(outputs_root: Path) -> pd.DataFrame:
    rows = []
    groups = [(t, p) for t in ("gl", "me", "pi") for p in ("ax", "co", "sa")]
    for p in outputs_root.glob("*/unet_effb0/groupdro/seed_*/training_summary.json"):
        ts = json.loads(p.read_text(encoding="utf-8"))
        q = ts.get("groupdro_q")
        if not q:
            continue
        parts = p.relative_to(outputs_root).parts
        study = parts[0]
        seed = int(parts[3].split("_")[-1])
        for (tumor, plane), w in zip(groups, q):
            rows.append({"study": study, "seed": seed, "tumor_code": tumor, "plane_code": plane, "weight": float(w)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate the complete paper experiment suite.")
    ap.add_argument("--results-root", default="results/paper_suite")
    ap.add_argument("--outputs-root", default="outputs/paper_suite")
    ap.add_argument("--output-dir", default="results/paper_suite/aggregate")
    ap.add_argument("--primary-architecture", default="unet_effb0")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    outputs_root = Path(args.outputs_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    group_rows: list[dict] = []
    hc3_rows: list[dict] = []
    raw: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    for p in sorted(results_root.glob("**/calibrated/per_case_metrics.csv")):
        parsed = parse_result_path(p, results_root)
        if parsed is None:
            continue
        study, arch, method, seed, _variant = parsed
        df = pd.read_csv(p)
        ts = training_summary(outputs_root, study, arch, method, seed)
        small_thr = ts.get("training_small_threshold_gt_pixels")
        sm, factor_rows = summarize_run(df, float(small_thr) if small_thr is not None else None)
        run_rows.append({"study": study, "architecture": arch, "method": method, "seed": seed, "small_threshold": small_thr, **sm})
        raw[(study, arch, method, seed)] = df
        for fr in factor_rows:
            hc3_rows.append({"study": study, "arch": arch, "method": method, "seed": seed, **fr})
        g = df.groupby(["tumor_code", "plane_code"])["dice"].agg(["mean", "std", "count"]).reset_index()
        for _, r in g.iterrows():
            group_rows.append(
                {
                    "study": study,
                    "architecture": arch,
                    "method": method,
                    "seed": seed,
                    "tumor_code": r.tumor_code,
                    "plane_code": r.plane_code,
                    "dice": r["mean"],
                    "within_seed_std": r["std"],
                    "n": int(r["count"]),
                }
            )

    if not run_rows:
        raise SystemExit(f"No calibrated result files found under {results_root}")
    run_df = pd.DataFrame(run_rows).sort_values(["study", "architecture", "method", "seed"])
    run_df.to_csv(out / "per_run_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "per_run_group_summary.csv", index=False)

    hc3_df = pd.DataFrame(hc3_rows)
    hc3_df.to_csv(out / "hc3_sensitivity_all_methods.csv", index=False)
    hc3_df[hc3_df.method == "erm"].to_csv(out / "hc3_sensitivity.csv", index=False)

    metric_cols = [
        "dice", "iou", "precision", "recall", "hd95", "assd", "p10_dice", "bottom20_dice",
        "failure_lt_0p8", "failure_lt_0p5", "failure_lt_0p1", "worst_group_dice", "group_gap",
        "gl_ax_dice", "small_gl_ax_dice", "small_worst_group_dice",
        "tumor_partial_eta2", "size_partial_eta2", "plane_partial_eta2", "interaction_partial_eta2",
        "tumor_p_classic", "size_p_classic", "plane_p_classic", "interaction_p_classic",
        "tumor_p_hc3", "size_p_hc3", "plane_p_hc3", "interaction_p_hc3",
    ]
    agg_rows = []
    for keys, sub in run_df.groupby(["study", "architecture", "method"]):
        row = {"study": keys[0], "architecture": keys[1], "method": keys[2], "seeds": int(sub.seed.nunique())}
        for c in metric_cols:
            vals = sub[c].astype(float).to_numpy()
            finite = np.isfinite(vals)
            row[c + "_mean"] = float(np.nanmean(vals)) if finite.any() else float("nan")
            row[c + "_std"] = float(np.nanstd(vals, ddof=1)) if finite.sum() > 1 else 0.0
        row["gl_ax_worst_seed_fraction"] = float(np.mean(sub.gl_ax_rank_from_worst.to_numpy(float) == 1.0))
        row["interaction_classic_significant_seed_fraction"] = float(np.mean(sub.interaction_p_classic.to_numpy(float) < 0.05))
        row["interaction_hc3_significant_seed_fraction"] = float(np.mean(sub.interaction_p_hc3.to_numpy(float) < 0.05))
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values(["study", "architecture", "method"])
    agg.to_csv(out / "multi_seed_summary.csv", index=False)

    recurrence = (
        run_df[(run_df.study == "standard") & (run_df.method == "erm")]
        .groupby("architecture")
        .agg(
            seeds=("seed", "nunique"),
            overall_dice_mean=("dice", "mean"),
            overall_dice_std=("dice", "std"),
            gl_ax_dice_mean=("gl_ax_dice", "mean"),
            gl_ax_dice_std=("gl_ax_dice", "std"),
            small_gl_ax_dice_mean=("small_gl_ax_dice", "mean"),
            small_gl_ax_dice_std=("small_gl_ax_dice", "std"),
            worst_group_dice_mean=("worst_group_dice", "mean"),
            gl_ax_worst_seed_fraction=("gl_ax_rank_from_worst", lambda x: float(np.mean(np.asarray(x) == 1.0))),
            interaction_classic_sig_fraction=("interaction_p_classic", lambda x: float(np.mean(np.asarray(x) < 0.05))),
            interaction_hc3_sig_fraction=("interaction_p_hc3", lambda x: float(np.mean(np.asarray(x) < 0.05))),
            interaction_partial_eta2_mean=("interaction_partial_eta2", "mean"),
        )
        .reset_index()
    )
    recurrence.to_csv(out / "architecture_recurrence.csv", index=False)

    paired = paired_tests(raw, run_df, args.primary_architecture)
    paired.to_csv(out / "paired_method_tests.csv", index=False)

    effects = effect_recurrence(run_df)
    effects.to_csv(out / "effect_recurrence.csv", index=False)

    qdf = groupdro_weights(outputs_root)
    qdf.to_csv(out / "groupdro_final_weights.csv", index=False)

    # Ready-to-copy main performance table for the paper.
    main = agg[agg.study == "standard"].copy()
    table_rows = []
    for _, r in main.iterrows():
        table_rows.append(
            {
                "Architecture": r.architecture,
                "Method": r.method,
                "Seeds": int(r.seeds),
                "Dice": fmt_pm(r.dice_mean, r.dice_std),
                "Worst-group Dice": fmt_pm(r.worst_group_dice_mean, r.worst_group_dice_std),
                "GL-AX Dice": fmt_pm(r.gl_ax_dice_mean, r.gl_ax_dice_std),
                "Small GL-AX Dice": fmt_pm(r.small_gl_ax_dice_mean, r.small_gl_ax_dice_std),
                "P10 Dice": fmt_pm(r.p10_dice_mean, r.p10_dice_std),
                "HD95": fmt_pm(r.hd95_mean, r.hd95_std, 2),
            }
        )
    paper_table = pd.DataFrame(table_rows)
    paper_table.to_csv(out / "paper_table_main.csv", index=False)

    lines = [
        "# BRISC2025 subgroup-robustness reproduction report",
        "",
        "Generated from validation-calibrated thresholds. The standard data split is fixed across model seeds.",
        "",
        "## Main multi-seed table",
        "",
        paper_table.to_markdown(index=False),
        "",
        "## Cross-architecture ERM recurrence",
        "",
        recurrence.to_markdown(index=False),
        "",
        "## Size-controlled effect recurrence (classical and HC3)",
        "",
        effects.to_markdown(index=False),
        "",
        "## Paired ERM-vs-intervention comparisons",
        "",
        paired.to_markdown(index=False) if len(paired) else "No complete paired comparisons.",
        "",
        "## Interpretation guardrails",
        "",
        "- The reproducible primary factor is tumor type; do not elevate the exploratory tumor-by-plane interaction to a universal claim.",
        "- The HC3 analysis is a sensitivity check for heteroskedastic residuals and is reported alongside classical Type-II ANOVA.",
        "- The locked confirmatory holdout is internal, not external validation.",
        "- GroupDRO conclusions apply to the evaluated configuration (eta=0.05), not to all possible GroupDRO tuning choices.",
    ]
    (out / "PAPER_SUITE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote aggregate paper results to {out}")
    print(paper_table.to_string(index=False))


if __name__ == "__main__":
    main()
