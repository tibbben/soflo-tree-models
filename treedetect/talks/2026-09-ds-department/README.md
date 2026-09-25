# UM Data Science department talk — September 2026

The 10-minute talk on the Gables and BCNP detection work, given with Tim and Ahsan.
Slides cover the pipeline, the evaluation problem, both sites' results, the AI
workflow behind the code, and next steps. Speaker notes carry the spoken script.

| File | What it is |
|---|---|
| `ds_department_talk_2026-09.pptx` | the deck (16 slides, speaker notes in each slide's notes pane) |
| `ds_department_talk_2026-09.pdf` | PDF export, for viewing without PowerPoint |
| `figures/audit_verdicts.jpg` | four audited false positives with the verdict I gave each one |
| `figures/bcnp_three_panel.jpg` | one 30 m window of `plot_9_3`: photo, census trees, YOLO detections |
| `make_figures.py` | regenerates both figures above |

Every other figure in the deck is a committed `reports/` PNG:
`label_qa.png`, `aoi_split.png`, `eval_sample.png`, `bcnp_georef/offset_rotation_sweep.png`,
`bcnp_georef/palm_test_plot_18_2.png`.

## Numbers in the deck, and where they come from

| Slide | Claim | Source |
|---|---|---|
| Step 1 | 1,197 / 1,703 / 4,088 label sources, 640 merged outlines | `src/s01_build_labels.py`, `reports/label_qa.png` |
| Step 2 | P 0.40 / R 0.39 at IoU 0.4; swap run F1 0.32 | `outputs/eval_summary.csv`, `outputs/swap/comparison.csv` |
| Step 3 | 94 / 3 / 3 audit tags; 43 gap vs 51 localization; 5.4% blind | `outputs/audit/fp_rescore_summary.txt`, `outputs/fp_diag/fp_diag_summary.txt`, `outputs/audit/fn_buckets.csv` |
| Step 4 | recall 8.0 / 22.4 / 49.8 vs random 0.5 / 4.1 / 17.8 | `outputs/gables_distance_eval.csv` |
| Campus outcomes | F1 0.420 / 0.573 / 0.660; palm probe 89.9% vs 54.6% | `outputs/gables_scratch/benchmark_3way.csv`, `outputs/species_probe/probe_metrics.csv` |
| Big Cypress | 2,576 census vs 4,212 detections; F1 11.3 / 11.0 vs random 10.0 / 9.6 | `outputs/bcnp_peak/eval_metrics.csv`, `outputs/bcnp_yolo/eval_metrics.csv` |
| Ruling out | recall flat ~17% by DBH; alignment sweep peaks at (0,0) | `outputs/bcnp_yolo/size_eval.csv`, `outputs/bcnp_georef/sweep_conventions.csv` |

## Regenerating the two talk figures

Needs the BCNP data (same as `s13`–`s18`) and the audit crops from `s10`:

```bash
cd treedetect
python src/s10_audit.py                              # writes outputs/audit/fp_crops/
python talks/2026-09-ds-department/make_figures.py   # BCNP_DIR=... to override ~/Downloads/bcnp
```

The script prints and skips whichever figure's inputs are missing, so it is safe to
run with only one of the two datasets present.
