# V13 – reproduction guide (only what has been run)

Start from `REPRODUCTION_GUIDE.md` (V12) for environment setup and the OULAD preparation steps. The following V13 scripts were run in this session, from `experimental_code/`, with `IQEL_RAW` pointing at the raw OULAD folder.

| Experiment | Script | Output |
|---|---|---|
| LOMO folds (E7) | `run_lomo.py` | `results/lomo/lomo_fold_metrics.csv`, `results/lomo/lomo_*.npy` |
| LOMO analysis | `analyze_lomo.py` | `results/lomo/lomo_overall_summary.csv`, `lomo_paired_comparisons.csv`, `lomo_per_module_summary.csv` |
| Adapted B4 (E8) | `b4_hqiat_adapted.py <weeks> <seeds>` | `results/b4/b4_results.csv`, `results/b4/b4_w*_s*.npy` |
| B4 paired analysis | `analyze_b4.py` | `results/b4/b4_comparison_summary.csv`, `b4_paired_comparisons.csv` |
| Wide-grid controls (E11) | `train_e6.py --stage tune/refit` with arms `A2-w,A2s-w,A4-ring-w` | `results/e6_tuning.csv`, `results/e6_final.csv` (appended) |
| Noise and shots (E10) | `e10_noise.py` | `results/e10/e10_noise_summary.csv`, `e10_noise_raw.csv` |
| Qubit scaling (E9) | `e9_qubit_scaling.py <n> <arm> <seeds>` | `results/e9/e9_results.csv`, `results/e9/e9_*.npy` |

Settings are copied from the 2014B selection results; no run re-tunes on 2014J. A partial run of `run_v13_chain.py` was interrupted twice by sandbox resets; it was resumed with its completed outputs skipped, and the `e10_noise.py` multinomial bug was fixed before the final E10 run.
