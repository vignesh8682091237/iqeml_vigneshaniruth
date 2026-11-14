# IQEL-NET Q1 upgrade: execution report

## 1. File discovery
| Requested file | Google Drive | Sandbox / project workspace |
|---|---|---|
| qsim.py, train_nn.py, run_trees.py, data_lib.py, analyze.py, test_qsim.py, prep_compact.py | not found | found, `outputs/quantum_experiments/` (backed up read-only) |
| final_auc_ci.csv, paired_diffs.csv, entanglement_diag.csv, nn_results_n4.csv, predictions.zip | not found | found, same folder |
| IQEL-NET_paper_draft_v10.docx | not found | found, `outputs/` |
| OULAD dataset | not found | found as ZIP files in `/mnt/user-data/uploads/` (7 CSV files, extracted outside the workspace) |
Drive was searched by exact filenames and by content; the only IQEL-related Drive files are the RAC slides, a reading-list sheet and a notes document. No separate "Claude project files" folder was accessible.

## 2. Workspace
`IQEL-NET_Q1_UPGRADE/` with `original_backup/` (read-only, checksums in `logs/`), `experimental_code/`, `configs/`, `data/compact/`, `results/`, `logs/`, `reports/`, `manuscript/`. No original file was modified.

## 3. Completed
- Raw-data reproduction: the compact tables and week4/8/12 feature arrays rebuilt from the raw OULAD CSVs are identical to the original arrays.
- **E6**: tuning on 2013B+2013J, selection on 2014B, refit on all training presentations, single evaluation on 2014J. 14 configurations x 3 weeks x 5 seeds (1 for logistic regression): 135 neural/quantum runs and 63 tree runs, all present.
- **E3**: four topologies (intra, pair, ring, all), verified against an independent PennyLane definition before training.
- **E1**: parameter-matched classical control A2s (16 fusion parameters, same as the ring circuit); also A1n (MLP, no bottleneck).
- **QECS validation**: original, centred, per-seed, sign-invariant, outcome comparison, correlation with entanglement entropy.
- **Statistics**: 1,000-resample student-level and module-clustered bootstrap, paired comparisons, Holm correction over 36 comparisons, per-module AUC, ECE, reliability diagrams, runtime.
- **Mismatches**: BERT, three heads, tuned loss weights and the old Figure 1 are fixed **by correcting the manuscript to the implementation** (BERT was not implemented and is stated as such); reliability diagrams now exist (Figure 3).
- Manuscript v11, README, requirements.txt, configuration file, unit-test log (20 passed).

## 4. Main results (AUC-ROC, 2014J, mean of 5 seeds; Weeks 4 / 8 / 12)
- Random forest, all features (A1): 0.742 / 0.778 / 0.796 (best).
- Classical bilinear fusion A2: 0.728 / 0.765 / 0.781. Ring circuit A4-ring: 0.718 / 0.753 / 0.758.
- Topology: intra 0.700 / 0.732 / 0.727; pair 0.713 / 0.753 / 0.770; ring as above; all 0.722 / 0.754 / 0.765. Cross-modal gates add 0.013 to 0.043 AUC over intra (student-level Holm p < 0.05 in all 9 comparisons; module-clustered significant in 6 of 9).
- Circuits stay below A2 (by 0.010, 0.011, 0.024 for the ring). The 16-parameter control A2s was unstable (seed SD up to 0.052), so the same-size comparison is inconclusive.
- QECS: predictive on its own (sign-invariant AUC 0.62 to 0.69) but its sign flips between seeds in 20% to 80% of seeds; centred QECS (exactly 0 for intra) has oriented AUC 0.57 to 0.66 in cross-modal circuits. No stable direction can be claimed. QECS and S_ent are essentially uncorrelated (|r| <= 0.28).
- Cost: circuits take 10 to 23 times longer per epoch than A2 (n = 4, one CPU core).

## 5. Errors encountered
- A shell brace-expansion slip created one stray folder name, removed.
- One unit test first failed because the raw-data path variable was not set (environment, not leakage); rerun with the variable set: 20/20 pass.
- The sandbox was paused twice and killed the background job; the pipeline is resumable and was relaunched. Logs of the interrupted segment were partly lost; all result files are complete and counted.
- Not tested on Windows (the sandbox is Linux). If JAX installation fails on Windows, use WSL2.

## 6. Pending / not done
BERT or any genuine text modality; leave-one-module-out; B4 (HQIAT-ML reproduction); n > 4; data re-uploading and richer readouts (E2, E4 proper); noise and shot sensitivity; second tuning seed and wider grids; regularisation tuning for A1n; verification of references [1]-[3]; systematic literature review; Q1 submission-readiness remains "not ready".

## 6a. Robustness targets (added after the main report)
T1 (Withdrawn vs all others) and T3 (Withdrawn vs Pass) were run with the identical protocol, five seeds, Weeks 4/8/12. Results: `results/e6_final_T1.csv`, `results/e6_final_T3.csv`, `results/trees_final_T1.csv`, `results/trees_final_T3.csv`, summaries in `results/robustness_T1_summary.csv` and `robustness_T3_summary.csv`. The cross-modal gain over the intra circuit and the classical-fusion advantage keep their direction under both targets. These are point estimates only: the full bootstrap/Holm procedure was not run for T1/T3. Manuscript v12 (Section 4.8) reports them.

## 7. Locations
Everything is under `/mnt/user-data/outputs/IQEL-NET_Q1_UPGRADE/`. Manuscript: `manuscript/IQEL-NET_paper_draft_v11.docx` (copy also at `/mnt/user-data/outputs/`). Results: `results/*.csv`, `results/preds/`, `results/figures/`. Code: `experimental_code/`. Logs: `logs/`.
