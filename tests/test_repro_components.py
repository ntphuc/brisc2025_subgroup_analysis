from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import BRISC2025Dataset
from src.statistics import ancova_with_hc3


def test_dataset_reads_paired_manifest(tmp_path: Path):
    root = tmp_path / "brisc2025"
    (root / "segmentation_task/train/images").mkdir(parents=True)
    (root / "segmentation_task/train/masks").mkdir(parents=True)
    img = np.zeros((32, 32), dtype=np.uint8)
    img[8:24, 8:24] = 120
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 255
    Image.fromarray(img).save(root / "segmentation_task/train/images/brisc2025_train_00001_gl_ax_t1.jpg")
    Image.fromarray(mask).save(root / "segmentation_task/train/masks/brisc2025_train_00001_gl_ax_t1.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {
            "case_id": "brisc2025_train_00001_gl_ax_t1",
            "split": "train",
            "image_path": "segmentation_task/train/images/brisc2025_train_00001_gl_ax_t1.jpg",
            "mask_path": "segmentation_task/train/masks/brisc2025_train_00001_gl_ax_t1.png",
            "tumor_code": "gl",
            "plane_code": "ax",
        }
    ]).to_csv(manifest, index=False)
    ds = BRISC2025Dataset(manifest, "train", root, image_size=64, augment=False, imagenet_norm=True)
    x, y, meta = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (1, 64, 64)
    assert y.sum() > 0
    assert meta["tumor_code"] == "gl"
    assert meta["plane_code"] == "ax"


def test_ancova_hc3_outputs_all_factors():
    rng = np.random.default_rng(123)
    rows = []
    for tumor in ("gl", "me", "pi"):
        for plane in ("ax", "co", "sa"):
            for _ in range(12):
                area = rng.integers(100, 2000)
                base = {"gl": 0.72, "me": 0.91, "pi": 0.84}[tumor]
                dice = np.clip(base + 0.025 * np.log1p(area) / 8 + rng.normal(0, 0.03), 0, 1)
                rows.append({"tumor_code": tumor, "plane_code": plane, "gt_pixels": area, "dice": dice})
    classic, hc3, result = ancova_with_hc3(pd.DataFrame(rows))
    assert set(result) == {"tumor", "plane", "interaction", "size"}
    assert "C(tumor_code)" in classic.index
    assert "C(tumor_code)" in hc3.index
    assert 0 <= result["tumor"]["p_hc3"] <= 1
