from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def status(v: bool) -> str:
    return "PASS" if v else "MISSING/WEAK"


def main() -> None:
    ap = argparse.ArgumentParser(description="Check whether the full experiment package needed by the paper exists.")
    ap.add_argument("--aggregate-dir", default="results/paper_suite/aggregate")
    ap.add_argument("--output", default="results/paper_suite/aggregate/PAPER_READINESS.md")
    args = ap.parse_args()
    root = Path(args.aggregate_dir)
    runs = pd.read_csv(root / "per_run_summary.csv")
    paired = pd.read_csv(root / "paired_method_tests.csv") if (root / "paired_method_tests.csv").exists() else pd.DataFrame()
    effects = pd.read_csv(root / "effect_recurrence.csv") if (root / "effect_recurrence.csv").exists() else pd.DataFrame()

    std_erm = runs[(runs.study == "standard") & (runs.method == "erm")]
    arch_seed_complete = (
        std_erm.groupby("architecture")["seed"].nunique().reindex(
            ["unet_effb0", "unetpp_effb0", "deeplabv3p_resnet50"], fill_value=0
        ).ge(3).all()
    )
    robust_complete = all(
        runs[(runs.study == "standard") & (runs.architecture == "unet_effb0") & (runs.method == m)]["seed"].nunique() >= 3
        for m in ["balanced", "groupdro"]
    )
    confirm_complete = all(
        runs[(runs.study == "confirmatory") & (runs.architecture == "unet_effb0") & (runs.method == m)]["seed"].nunique() >= 3
        for m in ["erm", "balanced", "groupdro"]
    )
    paired_complete = len(paired) >= 12 and set(paired.study.unique()) >= {"standard", "confirmatory"}
    hc3_complete = False
    tumor_reproducible = False
    interaction_tempered = False
    if len(effects):
        try:
            s_t = effects[(effects.study == "standard") & (effects.factor == "tumor")].iloc[0]
            c_t = effects[(effects.study == "confirmatory") & (effects.factor == "tumor")].iloc[0]
            s_i = effects[(effects.study == "standard") & (effects.factor == "interaction")].iloc[0]
            c_i = effects[(effects.study == "confirmatory") & (effects.factor == "interaction")].iloc[0]
            hc3_complete = int(s_t.hc3_significant_runs) == 9 and int(c_t.hc3_significant_runs) == 3
            tumor_reproducible = s_t.partial_eta2_mean > 0.10 and c_t.partial_eta2_mean > 0.10
            interaction_tempered = int(s_i.hc3_significant_runs) < int(s_i.classic_significant_runs) and int(c_i.hc3_significant_runs) == 0
        except Exception:
            pass

    checks = [
        ("3 architectures x 3 ERM seeds on the standard protocol", arch_seed_complete),
        ("Balanced and GroupDRO each have 3 standard seeds", robust_complete),
        ("Locked internal confirmatory ERM/Balanced/GroupDRO each have 3 seeds", confirm_complete),
        ("Paired all/GL-AX/small-GL-AX comparisons exist for both protocols", paired_complete),
        ("HC3 sensitivity analysis is complete", hc3_complete),
        ("Tumor-type effect remains large across standard and confirmatory protocols", tumor_reproducible),
        ("Interaction claim is appropriately tempered by HC3 and confirmatory results", interaction_tempered),
    ]
    ok = sum(bool(v) for _, v in checks)
    tier = "COMPLETE PAPER REPRODUCTION PACKAGE" if ok == len(checks) else "INCOMPLETE PAPER REPRODUCTION PACKAGE"
    lines = [
        "# Paper evidence readiness",
        "",
        f"**Status: {tier}**",
        "",
        "This checks experimental completeness, not conference acceptance.",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    lines += [f"| {name} | {status(v)} |" for name, v in checks]
    lines += [
        "",
        "## Claim guardrails",
        "",
        "- The robustly reproducible result is the tumor-type disparity, not a universal GL-AX interaction.",
        "- The confirmatory set is internal and must not be described as external validation.",
        "- GroupDRO conclusions apply only to the reported eta=0.05 configuration.",
    ]
    text = "\n".join(lines)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
