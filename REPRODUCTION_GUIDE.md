# REPRODUCTION_GUIDE – IQEL-NET Q1 upgrade

This guide reproduces the experiments reported in manuscript v12. It describes only what was actually run. Pending work is listed in section 10.

## 1. System requirements
- Python 3.12 (tested). Python 3.10+ is expected to work.
- CPU only. Tested on one CPU core, about 1 GB RAM for the neural/quantum runs (the 8-qubit statevector is 256 amplitudes per sample).
- Disk: about 60 MB of derived data and results; about 2 GB free during the run.
- Tested on Linux (Ubuntu 24.04 sandbox). **Windows has not been tested.** If JAX fails to install on Windows, use WSL2 (Ubuntu) and run the same commands.

## 2. Package versions (exact versions used)
See `logs/exact_versions_used.txt` and `requirements.txt`:
jax 0.12.0.1, jaxlib 0.12.0.1, numpy 2.4.4, pandas 3.0.2, pyarrow 26.0.0, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8, pennylane 0.45.1 (used only as a reference in tests), pytest 9.2.0, pillow 12.1.1.

## 3. Environment setup
```
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Dataset preparation
The OULAD dataset is **not included** in this package (see section 4a). Obtain it yourself:
1. Download the Open University Learning Analytics Dataset from https://analyse.kmi.open.ac.uk/open_dataset and accept its licence terms.
2. Extract the archive so that these seven files sit in one folder, for example `data/raw/`:
   `assessments.csv, courses.csv, studentAssessment.csv, studentInfo.csv, studentRegistration.csv, studentVle.csv, vle.csv`
3. Tell the code where they are:
   ```
   export IQEL_RAW=/path/to/folder/with/the/csv/files     # Windows: $env:IQEL_RAW="C:\...\raw"
   ```
4. Build the derived tables and the week-specific feature arrays (this reads the 450 MB `studentVle.csv` once):
   ```
   cd experimental_code
   python prep_compact.py
   python make_week_data.py
   ```
   Expected output files in `data/compact/`: `c_daily.parquet`, `c_act.parquet`, `c_assess.parquet`, `c_reg.parquet`, `week4.npz`, `week8.npz`, `week12.npz`.
   These were verified in the sandbox: rebuilt arrays were identical to the arrays used for the reported experiments.

### 4a. What is excluded from the ZIP, and why
The derived per-student tables (`data/compact/*`) and the week-specific arrays contain one row per student keyed by `id_student`. They are OULAD content re-keyed, and redistributing them would be redistributing the dataset, so they are not in the package. The raw CSVs are excluded for the same reason. Everything in `results/` is aggregate: no result file contains a student identifier, and each prediction file is one probability per held-out row without any identifier. Steps 4.1–4.4 rebuild the excluded files locally.

## 5. Reproducing the experiments
Run from `experimental_code/`. Each step is resumable: finished runs are skipped.

**E6 (presentation-held-out validation) together with E3 (topology ablation) and E1 (parameter-matched control):**
```
python run_all.py          # trees (tune_trees.py) and neural/quantum arms (train_e6.py), target T2
```
Expected outputs: `results/e6_tuning.csv`, `results/e6_final.csv`, `results/trees_tuning.csv`, `results/trees_final.csv`, `results/preds/*.npy`, `results/preds/*_diag.npz`.
Approximate cost on one CPU core: about 1.5 hours.

- E3 topology arms: `A4-intra`, `A4-pair`, `A4-ring`, `A4-all` in `train_e6.py`.
- E1 parameter-matched control: `A2s` (16 fusion parameters). The 16 parameters are the same as the ring circuit's fusion-stage parameters.

**Robustness targets T1 and T3 (optional):**
```
python run_targets.py      # writes *_T1.csv and *_T3.csv variants
```

## 6. QECS analysis
The per-seed QECS diagnostics are saved by `train_e6.py` (`results/preds/*_diag.npz`). The statistics, including the sign-invariant AUC, the centred correlator and the Hanley–McNeil lower bounds, are computed by:
```
python analyze_e6.py 1000  # 1000 bootstrap resamples; writes results/e6_qecs_validation.csv and results/e6_qecs_per_seed.csv
```

## 7. Statistics, figures and tables
```
python analyze_e6.py 1000  # main results, paired comparisons (Holm), per-module AUC, ECE, reliability figures
python make_figure1.py     # manuscript/figure1_qemf_architecture.png
```
Expected outputs: `results/e6_main_results.csv`, `results/e6_paired_comparisons.csv`, `results/e6_per_module_auc.csv`, `results/e6_qecs_validation.csv`, `results/figures/reliability_week{4,8,12}.png`.

## 8. Unit tests
```
cd experimental_code
python -m pytest tests -v
```
20 tests: simulator versus an independent PennyLane definition (all four topologies, n = 3 and 4), topology edge structure, separability of the intra circuit, parameter counts, split disjointness, preprocessing that never uses test rows, checkpoint leakage (features are unchanged when all later events are removed), and parameter-matching of A2s. The recorded result is in `logs/unit_tests.txt`.

## 9. Expected output filenames
- `results/e6_main_results.csv` — AUC, PR-AUC, F1, recall, Brier, ECE, runtime, parameter counts (per arm and week)
- `results/e6_paired_comparisons.csv` — paired differences with student and module-clustered intervals and Holm-adjusted p-values
- `results/e6_qecs_validation.csv` — QECS and centred-QECS validation by configuration and week
- `results/e6_per_module_auc.csv` — per-module AUC
- `results/robustness_T1_summary.csv`, `results/robustness_T3_summary.csv`
- `results/figures/reliability_week{4,8,12}.png`
- `manuscript/figure1_qemf_architecture.png`, `manuscript/figure3_reliability.png`

## 10. Known limitations and pending experiments (not done)
- Quantum simulation only (exact statevector, n = 4 qubits per register). No hardware, no noise, no shot sensitivity.
- The semantic input is built from VLE activity-type day counts, which come from the same click log as the behavioural input. **No BERT or other text encoder was used.** This was a design-time plan; it has not been implemented.
- Leave-one-module-out evaluation: **not run.**
- B4 (reproduction of HQIAT-ML under this protocol): **not run.**
- Qubit-count sensitivity beyond n = 4 and data re-uploading (E2) and richer readouts (E4): **not run.**
- Wide-grid controls A2s-w and A4-ring-w: code exists, **results not produced.**
- The T1/T3 robustness results are point estimates; the full bootstrap/Holm procedure was **not** applied to them.
- Reference list entries [1]–[3] have **not** been verified against the original sources.
- Only one tuning seed and a two-value learning-rate grid were used for the neural arms.
- Windows execution has **not** been tested.
