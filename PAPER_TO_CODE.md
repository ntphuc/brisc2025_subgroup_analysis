# Paper-to-code map

This file shows exactly where each experimental component in the paper is implemented.

| Paper component | Code / output |
|---|---|
| BRISC2025 paired dataset and 256x256 preprocessing | `src/dataset.py` |
| Standard 3343/590/860 split, data seed 42 | `scripts/prepare_brisc2025.py` |
| Locked internal 2757/588/588 confirmatory split | `scripts/make_confirmatory_manifest.py` |
| U-Net / EfficientNet-B0 | `src/model.py::UNetEfficientNetB0` |
| U-Net++ / EfficientNet-B0 | `src/model.py::UNetPlusPlusEfficientNetB0` |
| DeepLabV3+ / ResNet50 | `src/model.py::DeepLabV3PlusResNet50` |
| BCE + soft-Dice ERM | `src/paper_metrics.py`, `train_paper.py` |
| Inverse-frequency balanced sampling | `train_paper.py::_balanced_sampler` |
| GroupDRO eta=0.05 | `train_paper.py::_groupdro_loss` |
| Validation-only threshold calibration | `scripts/evaluate_paper_checkpoint.py::calibrate` |
| Dice / IoU / precision / recall / specificity | `src/paper_metrics.py::hard_batch_metrics` |
| HD95 / ASSD | `src/paper_metrics.py::surface_distances` |
| P10, bottom-20%, failures, worst-group | `src/paper_metrics.py::reliability_summary` and aggregation |
| Training-only q33/q67 lesion-size regimes | `train_paper.py`, `scripts/analyze_size_controlled.py` |
| Size-controlled ANCOVA | `src/statistics.py` |
| HC3 robust sensitivity | `src/statistics.py::ancova_with_hc3` |
| Paired bootstrap / Wilcoxon / Cohen dz | `scripts/aggregate_paper_suite.py` |
| Table 2 | `scripts/build_paper_artifacts.py` -> `table2_standard_performance.csv` |
| Table 3 | `scripts/build_paper_artifacts.py` -> `table3_effect_recurrence.csv` |
| Table 4 | `scripts/build_paper_artifacts.py` -> `table4_paired_deltas.csv` |
| Tumor x plane heatmaps | `scripts/build_paper_artifacts.py` -> `fig1_group_heatmaps.png` |
| Effect-size comparison | `scripts/build_paper_artifacts.py` -> `fig2_effect_sizes.png` |
| Mitigation trade-off | `scripts/build_paper_artifacts.py` -> `fig3_mitigation_tradeoff.png` |
| GroupDRO final q weights | `scripts/build_paper_artifacts.py` -> `fig4_groupdro_weights.png` |
| Archived paper aggregate values | `reference_results/` |
| Reproduction check | `scripts/verify_reproduction.py` |

## End-to-end entry points

Single workstation / GPU:

```bash
bash reproduce.sh strong
```

UIT Slurm:

```bash
sbatch run_uit.slurm
```

The final human-readable report is:

```text
results/paper_suite/aggregate/PAPER_SUITE_REPORT.md
```
