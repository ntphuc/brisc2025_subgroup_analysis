from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<case>brisc2025_(?P<official_split>train|test)_(?P<index>\d+)_(?P<tumor>gl|me|pi)_(?P<plane>ax|co|sa)_(?P<sequence>[^.]+))$",
    re.IGNORECASE,
)
GROUP_ORDER = [(t, p) for t in ("gl", "me", "pi") for p in ("ax", "co", "sa")]
EXPECTED_OFFICIAL_TRAIN = 3933
EXPECTED_OFFICIAL_TEST = 860


def find_brisc_root(path: Path) -> Path:
    candidates = [path, path / "brisc2025"]
    candidates.extend(p.parent for p in path.glob("**/segmentation_task") if p.is_dir())
    for c in candidates:
        if (c / "segmentation_task" / "train" / "images").is_dir() and (c / "segmentation_task" / "test" / "images").is_dir():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not find segmentation_task/train/images below {path}. "
        "Expected the official BRISC2025 directory layout."
    )


def scan_pairs(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    seg = root / "segmentation_task"
    for official_split in ("train", "test"):
        img_dir = seg / official_split / "images"
        mask_dir = seg / official_split / "masks"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing {img_dir} or {mask_dir}")
        for image_path in sorted(img_dir.iterdir(), key=lambda p: p.name):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = image_path.stem
            m = FILENAME_RE.match(stem)
            if m is None:
                raise ValueError(
                    f"Unexpected BRISC filename: {image_path.name}. "
                    "The paper relies on filename-encoded tumor and plane metadata."
                )
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                # Some mirrors may preserve the image extension for masks.
                alternatives = [p for p in mask_dir.glob(stem + ".*") if p.is_file()]
                if len(alternatives) == 1:
                    mask_path = alternatives[0]
                else:
                    raise FileNotFoundError(f"No unique mask found for {image_path}")
            rows.append(
                {
                    "case_id": stem,
                    "official_split": official_split,
                    "index": int(m.group("index")),
                    "tumor_code": m.group("tumor").lower(),
                    "plane_code": m.group("plane").lower(),
                    "sequence": m.group("sequence").lower(),
                    "image_path": image_path.relative_to(root).as_posix(),
                    "mask_path": mask_path.relative_to(root).as_posix(),
                }
            )
    return pd.DataFrame(rows)


def assign_standard_split(df: pd.DataFrame, seed: int, val_fraction: float) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out["official_split"]
    train = out[out["official_split"] == "train"].copy()
    rng = np.random.default_rng(int(seed))
    for tumor, plane in GROUP_ORDER:
        ids = train[(train.tumor_code == tumor) & (train.plane_code == plane)].index.to_numpy(dtype=int)
        # Keep the assignment independent of pandas/hash ordering.
        ids = np.sort(ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * float(val_fraction))))
        out.loc[ids[:n_val], "split"] = "val"
        out.loc[ids[n_val:], "split"] = "train"
    out.loc[out["official_split"] == "test", "split"] = "test"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the paired, paper-standard BRISC2025 manifest.")
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--output", default="data/brisc2025_manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--allow-nonstandard-counts", action="store_true")
    args = ap.parse_args()

    root = find_brisc_root(Path(args.data_root))
    df = scan_pairs(root)
    n_train = int((df.official_split == "train").sum())
    n_test = int((df.official_split == "test").sum())
    if not args.allow_nonstandard_counts and (n_train != EXPECTED_OFFICIAL_TRAIN or n_test != EXPECTED_OFFICIAL_TEST):
        raise SystemExit(
            f"Unexpected BRISC segmentation counts: train={n_train}, test={n_test}; "
            f"paper expects {EXPECTED_OFFICIAL_TRAIN}/{EXPECTED_OFFICIAL_TEST}. "
            "Use --allow-nonstandard-counts only for a deliberate non-paper smoke test."
        )

    out = assign_standard_split(df, args.seed, args.val_fraction)
    # Stable row order helps make the split reproducible across machines.
    out = out.sort_values(["official_split", "tumor_code", "plane_code", "index", "case_id"], kind="mergesort").reset_index(drop=True)
    cols = ["case_id", "split", "official_split", "index", "tumor_code", "plane_code", "sequence", "image_path", "mask_path"]
    out = out[cols]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    counts = out.groupby(["split", "tumor_code", "plane_code"]).size().reset_index(name="n")
    summary = {
        "data_root_resolved": str(root),
        "split_seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "counts_by_split": out.groupby("split").size().to_dict(),
        "counts_by_group": counts.to_dict(orient="records"),
    }
    (output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
