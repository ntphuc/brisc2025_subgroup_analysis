from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def case_column(df: pd.DataFrame) -> str:
    for c in ("case_id", "id", "sample_id"):
        if c in df.columns:
            return c
    raise ValueError("Manifest needs case_id/id/sample_id for protocol locking")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the fixed internal confirmatory holdout from development data only.")
    ap.add_argument("--input", default="data/brisc2025_manifest.csv")
    ap.add_argument("--output", default="data/brisc2025_manifest_confirmatory.csv")
    ap.add_argument("--lock", default="data/brisc2025_confirmatory_lock.json")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--confirmatory-fraction", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    out = Path(args.output)
    lock_path = Path(args.lock)
    input_sha = sha256_file(input_path)
    if out.exists() and lock_path.exists() and not args.force:
        old = json.loads(lock_path.read_text(encoding="utf-8"))
        same = (
            old.get("source_manifest_sha256") == input_sha
            and int(old.get("split_seed", -1)) == int(args.seed)
            and float(old.get("val_fraction", -1)) == float(args.val_fraction)
            and float(old.get("confirmatory_fraction", -1)) == float(args.confirmatory_fraction)
        )
        if not same:
            raise SystemExit(
                "Existing confirmatory lock does not match the current source manifest/parameters. "
                "For a deliberate fresh rebuild, delete the old data lock or pass --force and document the change."
            )
        print(f"REUSE locked confirmatory manifest: {out}")
        print(lock_path.read_text(encoding="utf-8"))
        return

    df = pd.read_csv(input_path)
    required = {"split", "tumor_code", "plane_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    case_col = case_column(df)

    original_test = df[df["split"].astype(str).str.lower() == "test"].copy()
    dev = df[df["split"].astype(str).str.lower().isin(["train", "val"])].copy()
    if len(dev) == 0:
        raise ValueError("No train/val development rows found")
    dev["split"] = ""
    rng = np.random.default_rng(args.seed)

    for (_, _), idx in dev.groupby(["tumor_code", "plane_code"], sort=True).groups.items():
        ids = np.asarray(sorted(idx), dtype=int)
        rng.shuffle(ids)
        n = len(ids)
        n_conf = max(1, int(round(n * args.confirmatory_fraction)))
        n_val = max(1, int(round(n * args.val_fraction)))
        if n_conf + n_val >= n:
            n_conf = max(1, n // 5)
            n_val = max(1, n // 5)
        dev.loc[ids[:n_conf], "split"] = "confirmatory"
        dev.loc[ids[n_conf:n_conf + n_val], "split"] = "val"
        dev.loc[ids[n_conf + n_val:], "split"] = "train"

    original_test["split"] = "exploratory_test"
    merged = pd.concat([dev, original_test], axis=0).sort_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    confirm_ids = sorted(merged.loc[merged["split"] == "confirmatory", case_col].astype(str).tolist())
    digest = hashlib.sha256("\n".join(confirm_ids).encode("utf-8")).hexdigest()
    counts = merged.groupby(["split", "tumor_code", "plane_code"]).size().reset_index(name="n")
    lock = {
        "source_manifest": str(input_path),
        "source_manifest_sha256": input_sha,
        "output_manifest": str(out),
        "split_seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "confirmatory_fraction": float(args.confirmatory_fraction),
        "confirmatory_n": len(confirm_ids),
        "confirmatory_case_ids_sha256": digest,
        "important_note": "Internal locked holdout from prior development train/val; it is not an external dataset and must not be described as external validation.",
        "counts": counts.to_dict(orient="records"),
    }
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
