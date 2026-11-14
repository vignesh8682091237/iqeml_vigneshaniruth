"""Runs the complete E6/E3/E1 pipeline sequentially (one process at a time, low RAM). Resumable: finished runs are skipped."""
import subprocess, sys
py = sys.executable
subprocess.run([py, "tune_trees.py"], check=True)
order = "A1n,A2,A2s,A4-ring,A4-intra,A4-pair,A4-all,A5,A3"
for wk in ("12", "8", "4"):
    subprocess.run([py, "train_e6.py", "--stage", "tune", "--arms", order, "--weeks", wk, "--lrs", "0.003,0.01"], check=True)
    subprocess.run([py, "train_e6.py", "--stage", "refit", "--arms", order, "--weeks", wk], check=True)
