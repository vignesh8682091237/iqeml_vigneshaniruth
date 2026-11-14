"""E9: qubit scaling at Week 12 (target T2, test 2014J). usage: python e9_qubit_scaling.py <n> <arm A4-ring|A2> <seeds> [probe]
Setting for each n: learning rate and epoch count are COPIED from the n=4 tuning on 2014B (no per-n tuning), so only n changes.
Records AUC, parameter counts, training seconds and peak process memory (maxrss). 'probe' only measures one training step for infeasible sizes."""
import sys, time, csv, resource, numpy as np, jax
import train_e6 as E, qsim_topo as T, data_lib
from sklearn.metrics import roc_auc_score
from paths import RESULTS
n = int(sys.argv[1]); arm = sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]; probe = len(sys.argv) > 4 and sys.argv[4] == "probe"
E.N_Q = n; E.init.__defaults__ = (n,); E.forward.__defaults__ = (n, False)
CFG = {"A4-ring": (0.003, 55), "A2": (0.01, 7)}; lr, epochs = CFG[arm]          # = selected on 2014B at n = 4 (results/e6_tuning.csv)
d = data_lib.load_week(12); y = d["y"].astype("int32"); tune_m, val_m, full_m, test_m = E.split_masks(d); fi, xi = np.where(full_m)[0], np.where(test_m)[0]
X = E.preprocess(d, full_m); sg = T.signs(n); out = RESULTS / "e9"; path = out / "e9_results.csv"
if probe:
    ds, db, dd = X["sem"].shape[1], X["beh"].shape[1], X["demo"].shape[1]; p = E.init(arm, jax.random.PRNGKey(0), ds, db, dd, n); m = jax.tree_util.tree_map(np.zeros_like, p); v = jax.tree_util.tree_map(np.zeros_like, p)
    j = fi[:64]; t0 = time.time(); E.step(p, m, v, 1, lr, X["sem"][j], X["beh"][j], X["demo"][j], y[j], sg, arm)[3].block_until_ready(); t1 = time.time() - t0
    t0 = time.time(); E.step(p, m, v, 2, lr, X["sem"][j], X["beh"][j], X["demo"][j], y[j], sg, arm)[3].block_until_ready(); t2 = time.time() - t0
    print(f"PROBE n={n} batch=64: compile+step {t1:.1f}s, steady step {t2:.2f}s, projected epoch ~{t2 * len(fi) / 64:.0f}s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
    sys.exit(0)
for seed in seeds:
    pf = out / f"e9_{arm}_n{n}_w12_s{seed}.npy"
    if pf.exists(): continue
    p, info = E.train(arm, X, y, fi, lr, seed, sg, epochs=epochs); pr = E.predict(arm, p, X, xi, sg)["prob"]; np.save(pf, pr.astype("float32"))
    row = dict(arm=arm, n=n, seed=seed, lr=lr, epochs=epochs, auc=roc_auc_score(y[xi], pr), secs_train=round(info["secs_train"], 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), **info["params"])
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); (new and w.writeheader()); w.writerow(row)
    print(arm, "n", n, "seed", seed, "AUC %.4f" % row["auc"], "%.0fs" % info["secs_train"], "rss %d MB" % row["maxrss_mb"], flush=True)
