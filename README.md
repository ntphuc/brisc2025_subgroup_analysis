# BRISC2025 Subgroup Robustness & Replicability

Official code package for the paper:

> **Beyond Average Dice: A Multi-Architecture Audit of Subgroup Robustness and Replicability in Brain-Tumor MRI Segmentation**

The repository is intentionally focused on the experiments reported in the paper.
It reproduces the three segmentation architectures, ERM / balanced sampling /
GroupDRO training, validation-only threshold calibration, subgroup/tail metrics,
size-controlled ANCOVA with HC3 sensitivity analysis, the locked internal
confirmatory protocol, paired bootstrap/Wilcoxon tests, and the paper tables and
figures.

Dataset: **BRISC2025** (Scientific Data, 2026), public at Kaggle under
`briscdataset/brisc2025`.

Paper code URL: <https://github.com/ntphuc/brisc2025_subgroup_analysis.git>

---

## 1. What this repository reproduces

The full `strong` profile runs **24 training jobs**:

| Study | Architecture | Method | Seeds | Runs |
|---|---|---|---|---:|
| Standard exploratory | U-Net / EfficientNet-B0 | ERM | 42, 43, 44 | 3 |
| Standard exploratory | U-Net++ / EfficientNet-B0 | ERM | 42, 43, 44 | 3 |
| Standard exploratory | DeepLabV3+ / ResNet50 | ERM | 42, 43, 44 | 3 |
| Standard exploratory | U-Net / EfficientNet-B0 | Balanced sampling | 42, 43, 44 | 3 |
| Standard exploratory | U-Net / EfficientNet-B0 | GroupDRO | 42, 43, 44 | 3 |
| Locked internal confirmatory | U-Net / EfficientNet-B0 | ERM | 42, 43, 44 | 3 |
| Locked internal confirmatory | U-Net / EfficientNet-B0 | Balanced sampling | 42, 43, 44 | 3 |
| Locked internal confirmatory | U-Net / EfficientNet-B0 | GroupDRO | 42, 43, 44 | 3 |

The standard protocol uses the paper split **3343 train / 590 validation / 860
test**. The confirmatory protocol uses **2757 train / 588 validation / 588 locked
internal confirmatory** cases and never evaluates on the already explored 860-case
test set.

The repository also regenerates:

- overall Dice, IoU, precision, recall, specificity, HD95 and ASSD;
- P10 Dice, bottom-20% Dice, failure rates at Dice < 0.8 / 0.5 / 0.1;
- the 3 x 3 tumor-type x imaging-plane grid;
- worst-group, GL-AX and small-GL-AX Dice;
- training-only q33/q67 lesion-size thresholds;
- `Dice ~ Tumor * Plane + log1p(LesionArea)` Type-II ANCOVA;
- HC3 heteroskedasticity-robust Type-II F-test sensitivity analysis;
- paired bootstrap 95% CIs, Wilcoxon tests and Cohen's dz for ERM vs interventions;
- final GroupDRO adversarial weights;
- the CSV tables and PNG figures used to build the paper.

---

## 2. Installation

### 2.1 Clone

```bash
git clone https://github.com/ntphuc/brisc2025_subgroup_analysis.git
cd brisc2025_subgroup_analysis
```

### 2.2 Create a Python environment

Python 3.10-3.13 is supported by the code. A CUDA-capable PyTorch build is
recommended for the full experiment suite.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **PyTorch + Torchvision for your CUDA/CPU platform** using the command
recommended by <https://pytorch.org/get-started/locally/>, then install the
remaining dependencies:

```bash
pip install -r requirements.txt
```

Example CPU-only smoke-test environment:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

Verify the code before downloading/training:

```bash
pytest -q
```

---

## 3. Download BRISC2025

### Option A — automatic public Kaggle download

```bash
python scripts/download_brisc2025.py
```

The script uses `kagglehub` and exposes the dataset at:

```text
datasets/brisc2025/
  segmentation_task/
    train/images/
    train/masks/
    test/images/
    test/masks/
```

### Option B — manual download

Download `briscdataset/brisc2025` from Kaggle and either place it at
`datasets/brisc2025` or set:

```bash
export BRISC_DATA_ROOT=/absolute/path/to/brisc2025
```

The BRISC root must contain `segmentation_task/`.

---

## 4. Prepare and verify the exact paper split

```bash
bash scripts/prepare_data.sh
```

This command creates:

```text
data/brisc2025_manifest_paper_standard.csv
data/brisc2025_manifest_confirmatory.csv
data/brisc2025_confirmatory_lock.json
```

and verifies the paper counts. The standard split is stratified by the nine
`tumor_code x plane_code` groups with **data split seed 42**. Model seeds do not
change sample membership.

Expected standard counts:

```text
train = 3343
val   = 590
test  = 860
```

Expected confirmatory counts:

```text
train         = 2757
val           = 588
confirmatory  = 588
explored test = 860  (renamed exploratory_test and inaccessible to confirmatory evaluation)
```

If these counts do not match, the script stops instead of silently running a
non-paper protocol.

---

## 5. Fast sanity check

Before spending the full compute budget, run the unit tests:

```bash
pytest -q
```

Then run the `quick` experiment profile (U-Net seed 42 with ERM, Balanced and
GroupDRO):

```bash
bash reproduce.sh quick
```

`quick` checks the full train -> validation threshold calibration -> test
evaluation -> statistical aggregation path. It is **not** the full paper result.

---

## 6. Reproduce the complete paper on one machine

On a workstation/server with one GPU, run:

```bash
bash reproduce.sh strong
```

This executes all 24 runs sequentially and then automatically runs aggregation,
HC3 analysis, paired statistics, readiness checks, reference-result comparison,
and paper-artifact generation.

Other profiles:

```bash
bash reproduce.sh core    # 15 standard-study runs, no confirmatory study
bash reproduce.sh quick   # 3 seed-42 U-Net sanity runs
```

Because the `strong` profile trains 24 models, a Slurm cluster is substantially
faster than a single GPU.

---

## 7. Reproduce on the authors' UIT Slurm cluster

After installation and dataset preparation, only the launcher is needed:

```bash
sbatch run_uit.slurm
```

The launcher submits the 24 GPU workers and one dependent CPU aggregation job.
Monitor with:

```bash
squeue -u "$USER"
```

For a quick cluster check:

```bash
PAPER_PROFILE=quick sbatch run_uit.slurm
```

`run_uit.slurm`, `run_paper_worker.slurm`, and `run_paper_aggregate.slurm` contain
UIT-specific `--account`, `--partition`, and GPU-resource directives. Researchers
on another Slurm cluster should edit only those scheduler headers; the experiment
commands and protocol remain unchanged.

---

## 8. Main outputs

Training checkpoints and logs:

```text
outputs/paper_suite/<study>/<architecture>/<method>/seed_<seed>/
```

Per-case evaluation and analysis:

```text
results/paper_suite/<study>/<architecture>/<method>/seed_<seed>/
```

Final aggregate outputs:

```text
results/paper_suite/aggregate/
  PAPER_SUITE_REPORT.md
  per_run_summary.csv
  per_run_group_summary.csv
  multi_seed_summary.csv
  architecture_recurrence.csv
  effect_recurrence.csv
  hc3_sensitivity.csv
  paired_method_tests.csv
  groupdro_final_weights.csv
  paper_table_main.csv
```

Paper tables and figures:

```text
results/paper_suite/paper_artifacts/
  table2_standard_performance.csv
  table3_effect_recurrence.csv
  table4_paired_deltas.csv
  fig1_group_heatmaps.png
  fig2_effect_sizes.png
  fig3_mitigation_tradeoff.png
  fig4_groupdro_weights.png
```

---

## 9. Expected paper-level results

The archived aggregate results used by the manuscript are in
`reference_results/`.

Important targets are approximately:

| Result | Paper value |
|---|---:|
| Standard U-Net/Eff-B0 ERM Dice | 0.8744 ± 0.0017 |
| Standard U-Net++/Eff-B0 ERM Dice | 0.8772 ± 0.0040 |
| Standard DeepLabV3+/R50 ERM Dice | 0.8733 ± 0.0018 |
| Standard U-Net GL-AX Dice | 0.7249 ± 0.0164 |
| Standard U-Net small GL-AX Dice | 0.5315 ± 0.0317 |
| Confirmatory U-Net ERM Dice | 0.8866 ± 0.0025 |
| Standard tumor partial eta² | 0.1474 |
| Confirmatory tumor partial eta² | 0.1612 |

For the standard nine ERM architecture-seed runs, the tumor-by-plane interaction
is significant in **8/9** classical Type-II tests but only **2/9** HC3-robust
sensitivity tests. On the locked confirmatory U-Net study it is significant in
**0/3** under both tests.

After a complete run:

```bash
python scripts/verify_reproduction.py
```

The verifier uses practical numeric tolerances because exact floating-point results
can vary with GPU architecture, CUDA, cuDNN, PyTorch and Torchvision versions.
It does not pretend that deep-learning training is bitwise reproducible across
hardware stacks.

---

## 10. Exact training settings

The paper configuration is in `configs/paper_suite.yaml`.

Core settings:

```text
image size            256 x 256
batch size            8
epochs                 <= 30
optimizer              AdamW
weight decay           1e-4
EfficientNet LR        3e-4
DeepLab/ResNet50 LR    1e-4
scheduler              cosine annealing
checkpoint             best validation mean Dice
patience               8
loss                   0.5 BCE + 0.5 soft Dice
model seeds            42, 43, 44
data split seed        42
confirmatory seed      20260926
GroupDRO eta           0.05
```

Training augmentation applies a paired horizontal flip, rotations up to ±10
degrees, and moderate image-only contrast/shift/gamma perturbations. Evaluation
uses no augmentation.

The binary threshold is selected **only on validation data**, searching 0.30 to
0.70 in steps of 0.025, then frozen for the evaluation split.

---

## 11. Method definitions

### ERM

Ordinary empirical risk minimization with per-image
`0.5 * BCE + 0.5 * soft-Dice loss`.

### Balanced sampling

Inverse-frequency weighted sampling over the nine tumor-plane groups. The loss is
unchanged; only sample exposure is rebalanced.

### GroupDRO

For group loss `L_g`, adversarial weights are updated as:

```text
q_g <- q_g * exp(eta * L_g)
q   <- q / sum(q)
```

with `eta = 0.05` in the reported experiment. The paper's conclusion about the
observed GroupDRO weight collapse applies to this evaluated configuration only.

---

## 12. Statistical protocol

For every ERM run the paper fits:

```text
Dice ~ C(tumor_code) * C(plane_code) + log1p(gt_pixels)
```

and reports:

1. classical Type-II ANOVA F tests;
2. partial eta-squared effect sizes;
3. HC3 heteroskedasticity-robust Type-II F-test sensitivity results.

For intervention comparisons, predictions are averaged across the three model
seeds for each case before computing paired method-minus-ERM Dice differences.
The repository reports 10,000 paired bootstrap resamples for the 95% CI,
two-sided Wilcoxon signed-rank tests, and paired Cohen's `d_z`.

---

## 13. Reproducibility notes and limitations

- The confirmatory split is an **internal locked holdout**, not an external dataset.
- The public BRISC representation does not provide a complete patient/study ID
  suitable for proving subject-level independence across every original source.
- HD95 and ASSD are reported in pixels because physical spacing is not
  consistently available.
- The full paper reports three model seeds, not a claim of exhaustive seed
  uncertainty.
- Deep-learning runs can differ slightly across CUDA/cuDNN/PyTorch versions.
- The code does not download or redistribute BRISC images inside this repository;
  users obtain the public dataset from its official source.

---

## 14. Repository structure

```text
configs/paper_suite.yaml          paper hyperparameters
src/dataset.py                    paired BRISC dataset + augmentation
src/model.py                      3 segmentation architectures
src/paper_metrics.py              overlap, tail and surface metrics
src/statistics.py                 Type-II ANCOVA + HC3 sensitivity
train_paper.py                    ERM / Balanced / GroupDRO training
scripts/download_brisc2025.py     optional public dataset downloader
scripts/prepare_brisc2025.py      exact standard split builder
scripts/make_confirmatory_manifest.py
scripts/evaluate_paper_checkpoint.py
scripts/analyze_size_controlled.py
scripts/aggregate_paper_suite.py
scripts/build_paper_artifacts.py
scripts/verify_reproduction.py
reproduce.sh                      one-machine launcher
run_uit.slurm                     UIT Slurm launcher
reference_results/                archived paper aggregate values
```

---

## 15. Citation

If this code is useful, please cite the accompanying ICEIR 2026 paper and the
BRISC2025 dataset paper. A machine-readable citation template is included in
`CITATION.cff`.
