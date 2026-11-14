"""Sequential, resumable chain: E11 wide-grid controls, E10 noise, E9 qubit scaling."""
import subprocess, sys
py = sys.executable; arms = "A2-w,A2s-w,A4-ring-w"
for wk in ("12", "8", "4"):
    subprocess.run([py, "train_e6.py", "--stage", "tune", "--arms", arms, "--weeks", wk, "--lrs", "0.0003,0.001,0.003,0.01,0.03"], check=True)
    subprocess.run([py, "train_e6.py", "--stage", "refit", "--arms", arms, "--weeks", wk], check=True)
subprocess.run([py, "e10_noise.py"], check=True)
for n, seeds in ((2, "0,1,2,3,4"), (4, "0,1,2"), (6, "0,1,2")):
    for arm in ("A2", "A4-ring"): subprocess.run([py, "e9_qubit_scaling.py", str(n), arm, seeds], check=True)
subprocess.run([py, "e9_qubit_scaling.py", "8", "A4-ring", "0", "probe"], check=False)
print("CHAIN DONE")
