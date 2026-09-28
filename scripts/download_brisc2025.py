from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


HANDLE = "briscdataset/brisc2025"


def find_dataset_root(root: Path) -> Path | None:
    root = root.resolve()
    candidates = [root, root / "brisc2025"]
    candidates.extend(p for p in root.glob("**/brisc2025") if p.is_dir())
    candidates.extend(p.parent for p in root.glob("**/segmentation_task") if p.is_dir())
    seen = set()
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            continue
        if p in seen:
            continue
        seen.add(p)
        if (p / "segmentation_task").is_dir():
            return p
    return None


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
    else:
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            print("Symlink creation failed; falling back to a physical copy.")
            shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download the public BRISC2025 Kaggle dataset with kagglehub and expose a stable local data root."
    )
    ap.add_argument("--output", default="datasets/brisc2025", help="Stable dataset path used by this repository")
    ap.add_argument("--copy", action="store_true", help="Copy the dataset instead of creating a symlink to the kagglehub cache")
    ap.add_argument("--force", action="store_true", help="Replace an existing output symlink/directory")
    args = ap.parse_args()

    out = Path(args.output)
    existing = find_dataset_root(out) if out.exists() else None
    if existing is not None and not args.force:
        print(f"BRISC2025 is already available at: {existing}")
        return
    if args.force and (out.exists() or out.is_symlink()):
        if out.is_symlink() or out.is_file():
            out.unlink()
        else:
            shutil.rmtree(out)

    try:
        import kagglehub
    except Exception as exc:
        raise SystemExit(
            "kagglehub is not installed. Run `pip install -r requirements.txt` first, "
            f"or download {HANDLE} manually and place it at {out}. Original error: {exc}"
        )

    print(f"Downloading public Kaggle dataset: {HANDLE}")
    cache_path = Path(kagglehub.dataset_download(HANDLE))
    root = find_dataset_root(cache_path)
    if root is None:
        raise SystemExit(
            f"Download completed at {cache_path}, but no directory containing segmentation_task was found. "
            "Inspect the Kaggle dataset layout and pass the actual BRISC root to --data-root when preparing the manifest."
        )
    print(f"Downloaded BRISC root: {root}")
    link_or_copy(root, out, copy=args.copy)
    print(f"Repository dataset path: {out}")


if __name__ == "__main__":
    main()
