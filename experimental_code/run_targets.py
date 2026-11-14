"""Robustness targets T1 and T3 (same protocol as T2). Resumable. One process at a time."""
import subprocess, sys
py = sys.executable
for tgt in ("T1", "T3"):
    subprocess.run([py, "tune_trees.py", "--target", tgt], check=True)
    for wk in ("12", "8", "4"):
        subprocess.run([py, "train_e6.py", "--stage", "tune", "--arms", "A1n,A2,A2s,A4-ring,A4-intra,A4-pair,A4-all,A5,A3", "--weeks", wk, "--lrs", "0.003,0.01", "--target", tgt], check=True)
        subprocess.run([py, "train_e6.py", "--stage", "refit", "--arms", "A1n,A2,A2s,A4-ring,A4-intra,A4-pair,A4-all,A5,A3", "--weeks", wk, "--target", tgt], check=True)
print("TARGETS DONE")
