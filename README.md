# IQEL-NET_Q1_UPGRADE

Experiments E6 (presentation-held-out validation), E3 (quantum topology ablation) and E1 (parameter-matched classical control) for the IQEL-NET manuscript.
All original files are untouched in `original_backup/` (read-only copy, checksums in `logs/original_backup_checksums.txt`).

## Layout
| Folder | Content |
|---|---|
| `original_backup/` | Original v10 manuscript, code and results exactly as before this upgrade |
| `experimental_code/` | New code (`train_e6.py`, `qsim_topo.py`, `tune_trees.py`, `analyze_e6.py`, `run_all.py`, tests) |
| `configs/e6_config.json` | Every setting used (generated from the code constants) |
| `data/` | `compact/` = derived per-student tables and `week{4,8,12}.npz` feature arrays (the raw OULAD CSVs are NOT included, see below) |
| `results/` | All result tables, `preds/` (every test prediction and QECS diagnostics), `figures/` |
| `logs/` | Run logs, unit-test log, exact package versions |
| `manuscript/` | Draft v11 and Figure 1 |
| `reports/` | Written report of this upgrade |

## Setup (Windows, CPU only)
1. Install Python 3.12 (3.10 or newer should work). In PowerShell: `py -m venv .venv ; .\.venv\Scripts\Activate.ps1`
2. `pip install -r requirements.txt`  (JAX CPU wheels exist for Windows; if installation fails, use WSL2 Ubuntu, where the same command works).
3. Download OULAD (https://analyse.kmi.open.ac.uk/open_dataset) and put the seven CSV files in `data/raw/` (or set `IQEL_RAW` to their folder).
   Only `studentVle.csv` (about 450 MB) is large; it is needed once, for step 4.

## Run
```
cd experimental_code
python prep_compact.py          # raw CSV -> compact tables (once)
python make_week_data.py        # compact tables -> week4/8/12.npz
python -m pytest tests -v       # 20 tests; first three groups need only the compact data and PennyLane
python run_all.py               # trees + all neural/quantum arms: tuning on 2014B, refit, single evaluation on 2014J
python analyze_e6.py 1000       # bootstrap, Holm, per-module AUC, calibration, QECS validation
python make_figure1.py
python run_targets.py          # optional: robustness targets T1 and T3
```
`run_all.py` is resumable (finished runs are skipped) and runs one process at a time to keep RAM below 1 GB.
Measured in the sandbox used here (one CPU core): about 1.5 hours for `run_all.py`, about 15 minutes for the analysis. A typical laptop should be faster.

## Protocol (do not change without re-running everything)
Tune on 2013B+2013J, select on 2014B, refit on 2013B+2013J+2014B, evaluate once on 2014J. The test presentation never enters tuning
(asserted in code and covered by `tests/test_protocol.py`).

## Not done / known limits
See `reports/Q1_upgrade_report.md` and manuscript Section 4.6.
