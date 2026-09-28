from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

EXPECTED_STANDARD = {"train": 3343, "val": 590, "test": 860}
EXPECTED_CONFIRMATORY = {"train": 2757, "val": 588, "confirmatory": 588, "exploratory_test": 860}
EXPECTED_STANDARD_TRAIN_GROUPS = {
    "gl-ax": 335, "gl-co": 365, "gl-sa": 275,
    "me-ax": 360, "me-co": 362, "me-sa": 408,
    "pi-ax": 362, "pi-co": 433, "pi-sa": 443,
}


def counts(path: str) -> tuple[dict, dict]:
    df = pd.read_csv(path)
    split = {str(k): int(v) for k, v in df.groupby("split").size().to_dict().items()}
    tr = df[df.split == "train"]
    groups = {f"{t}-{p}": int(n) for (t, p), n in tr.groupby(["tumor_code", "plane_code"]).size().items()}
    return split, groups


def main() -> None:
    std, groups = counts("data/brisc2025_manifest_paper_standard.csv")
    conf, _ = counts("data/brisc2025_manifest_confirmatory.csv")
    problems = []
    for k, v in EXPECTED_STANDARD.items():
        if std.get(k) != v:
            problems.append(f"standard {k}: expected {v}, got {std.get(k)}")
    for k, v in EXPECTED_CONFIRMATORY.items():
        if conf.get(k) != v:
            problems.append(f"confirmatory {k}: expected {v}, got {conf.get(k)}")
    for k, v in EXPECTED_STANDARD_TRAIN_GROUPS.items():
        if groups.get(k) != v:
            problems.append(f"standard train group {k}: expected {v}, got {groups.get(k)}")
    if problems:
        raise SystemExit("Paper protocol mismatch:\n- " + "\n- ".join(problems))
    print(json.dumps({"standard": std, "confirmatory": conf, "status": "paper protocol counts match"}, indent=2))


if __name__ == "__main__":
    main()
