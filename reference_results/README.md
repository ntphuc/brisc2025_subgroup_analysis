# Reference results from the paper run

These CSV files archive the aggregate outputs used to write the paper. They are
provided so a fresh reproduction can be checked without distributing large model
checkpoints or the BRISC2025 dataset itself.

Key paper targets include:

- Standard U-Net/EfficientNet-B0 ERM Dice: **0.8744 ± 0.0017**.
- Standard GL-AX Dice: **0.7249 ± 0.0164**.
- Standard small GL-AX Dice: **0.5315 ± 0.0317**.
- Standard U-Net++/Eff-B0 ERM Dice: **0.8772 ± 0.0040**.
- Standard DeepLabV3+/ResNet50 ERM Dice: **0.8733 ± 0.0018**.
- Locked internal confirmatory U-Net/Eff-B0 ERM Dice: **0.8866 ± 0.0025**.
- Mean partial eta-squared for tumor type: **0.1474** (standard) and **0.1612** (confirmatory).
- Tumor-by-plane interaction: classical Type-II ANOVA significant in **8/9** standard ERM runs but HC3 significant in only **2/9**; **0/3** under both tests on the locked confirmatory study.

Exact floating-point equality is not expected across GPU, CUDA, PyTorch and
Torchvision versions. `scripts/verify_reproduction.py` therefore uses practical
numeric tolerances and reports warnings rather than claiming bitwise reproducibility.
