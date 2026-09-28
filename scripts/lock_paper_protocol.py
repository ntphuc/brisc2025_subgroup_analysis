from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the experiment/analysis protocol before confirmatory runs.")
    ap.add_argument("--profile", choices=["quick", "core", "strong"], required=True)
    ap.add_argument("--seeds", default="42 43 44")
    ap.add_argument("--architectures", default="unet_effb0 unetpp_effb0 deeplabv3p_resnet50")
    ap.add_argument("--primary-architecture", default="unet_effb0")
    ap.add_argument("--data-split-seed", type=int, default=42)
    ap.add_argument("--confirmatory-split-seed", type=int, default=20260926)
    ap.add_argument("--output", default="results/paper_suite/PROTOCOL_LOCK.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = [
        "configs/paper_suite.yaml", "train_paper.py", "src/model.py", "src/dataset.py",
        "src/paper_metrics.py", "src/statistics.py", "scripts/prepare_brisc2025.py",
        "scripts/make_confirmatory_manifest.py", "scripts/evaluate_paper_checkpoint.py",
        "scripts/analyze_size_controlled.py", "scripts/aggregate_paper_suite.py",
        "scripts/build_paper_artifacts.py", "scripts/run_paper_task.sh", "reproduce.sh",
    ]
    hashes = {}
    for name in files:
        p = Path(name)
        if not p.exists():
            raise SystemExit(f"Protocol file is missing: {p}")
        hashes[name] = sha256_file(p)

    plan = {
        "requested_profile": args.profile,
        "profile_definition": "quick/core/strong are compute subsets of the same frozen scientific protocol",
        "model_seeds": [int(x) for x in args.seeds.split()],
        "architectures": args.architectures.split(),
        "primary_architecture": args.primary_architecture,
        "standard_methods": ["erm", "balanced", "groupdro"],
        "data_split_seed": args.data_split_seed,
        "confirmatory_split_seed": args.confirmatory_split_seed,
        "confirmatory_policy": "strong profile only: internal 15% holdout from prior development pool; previously explored official test renamed exploratory_test",
        "selection_policy": "best checkpoint chosen only by validation mean Dice; one global threshold selected only on validation mean Dice",
        "size_policy": "q33/q67 lesion-size thresholds computed from non-augmented training masks at 256x256",
        "statistics_policy": "mean+SD across three seeds; subgroup/tail metrics; OLS Type-II ANCOVA with HC3 sensitivity; paired bootstrap CI, Wilcoxon and Cohen dz after per-case seed averaging",
        "file_hashes": hashes,
    }
    hash_plan = dict(plan)
    hash_plan.pop("requested_profile", None)
    payload = json.dumps(hash_plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    plan["protocol_sha256"] = hashlib.sha256(payload).hexdigest()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != plan["protocol_sha256"]:
            raise SystemExit(
                f"Protocol lock already exists with a different hash: {out}. "
                "Delete results/paper_suite for a fresh reproduction or use --force only for a documented amendment."
            )
        print(f"Protocol already locked and unchanged: {plan['protocol_sha256']}")
        return
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Locked paper protocol: {plan['protocol_sha256']} -> {out}")


if __name__ == "__main__":
    main()
