from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

FORMULA = "dice ~ C(tumor_code) * C(plane_code) + log_gt_pixels"
TERM_MAP = {
    "tumor": "C(tumor_code)",
    "plane": "C(plane_code)",
    "interaction": "C(tumor_code):C(plane_code)",
    "size": "log_gt_pixels",
}


def prepare_ancova_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d[(d["gt_pixels"].astype(float) > 0) & d["dice"].notna()].copy()
    d["tumor_code"] = d["tumor_code"].astype(str).str.lower()
    d["plane_code"] = d["plane_code"].astype(str).str.lower()
    d["log_gt_pixels"] = np.log1p(d["gt_pixels"].astype(float))
    return d


def ancova_with_hc3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit the paper ANCOVA and return classical Type-II + HC3 Type-II tables.

    Partial eta^2 is computed from the classical Type-II sums of squares. HC3 is
    used as a heteroskedasticity-robust sensitivity analysis for the F tests.
    """
    d = prepare_ancova_frame(df)
    model = smf.ols(FORMULA, data=d).fit()
    classic = anova_lm(model, typ=2)
    hc3 = anova_lm(model, typ=2, robust="hc3")

    ss_e = float(classic.loc["Residual", "sum_sq"])
    result: dict[str, dict[str, float]] = {}
    for short, term in TERM_MAP.items():
        ss = float(classic.loc[term, "sum_sq"])
        eta = ss / (ss + ss_e) if ss + ss_e > 0 else float("nan")
        result[short] = {
            "partial_eta2": eta,
            "F_classic": float(classic.loc[term, "F"]),
            "p_classic": float(classic.loc[term, "PR(>F)"]),
            "F_hc3": float(hc3.loc[term, "F"]),
            "p_hc3": float(hc3.loc[term, "PR(>F)"]),
        }
    return classic, hc3, result
