"""B4 (adapted HQIAT-ML) vs A1, A2, A4-ring on the untouched 2014J test presentation. Paired module-clustered and student-level bootstrap, Holm correction."""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from paths import RESULTS
import data_lib
from train_e6 import split_masks
R = 1000; rng = np.random.RandomState(13)
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.
    for k, i in enumerate(o): run = max(run, (len(p) - k) * p[i]); adj[i] = min(1., run)
    return adj
FILES = {"B4": lambda w, s: RESULTS / "b4" / f"b4_w{w}_s{s}.npy", "A1": lambda w, s: RESULTS / "preds" / f"trees_A1_RF_w{w}_s{s}.npy",
         "A2": lambda w, s: RESULTS / "preds" / f"A2_w{w}_s{s}.npy", "A4-ring": lambda w, s: RESULTS / "preds" / f"A4-ring_w{w}_s{s}.npy"}
rows, prow = [], []
for wk in (4, 8, 12):
    d = data_lib.load_week(wk); _, _, _, tm = split_masks(d); y = d["y"][tm].astype(int); mod = d["module"][tm]; mods = np.unique(mod); n = len(y)
    P = {a: [np.load(f(wk, s)) for s in range(5)] for a, f in FILES.items()}
    for a, ps in P.items(): assert all(len(p) == n for p in ps), (a, wk)
    sidx = rng.randint(0, n, (R, n)); rows_of = {m: np.where(mod == m)[0] for m in mods}
    cidx = [np.concatenate([rows_of[m] for m in rng.choice(mods, len(mods))]) for _ in range(R)]
    boots = {}
    for a, ps in P.items():
        pt = np.array([auc(y, p) for p in ps]); bs = np.array([np.mean([auc(y[i], p[i]) for p in ps]) for i in sidx]); bc = np.array([np.mean([auc(y[i], p[i]) for p in ps]) for i in cidx])
        boots[a] = (pt, bs, bc); rows.append(dict(week=wk, arm=a, auc=pt.mean(), seed_sd=pt.std(ddof=1), ci_student_lo=np.percentile(bs, 2.5), ci_student_hi=np.percentile(bs, 97.5), ci_module_lo=np.percentile(bc, 2.5), ci_module_hi=np.percentile(bc, 97.5)))
    for a, b in (("B4", "A1"), ("B4", "A2"), ("B4", "A4-ring")):
        r = dict(week=wk, a=a, b=b, diff=boots[a][0].mean() - boots[b][0].mean())
        for tag, k in (("student", 1), ("module", 2)):
            dd = boots[a][k] - boots[b][k]; r[f"ci_{tag}_lo"], r[f"ci_{tag}_hi"] = np.percentile(dd, [2.5, 97.5]); r[f"p_{tag}"] = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1. / (R + 1))
        prow.append(r)
M = pd.DataFrame(rows).round(4); M.to_csv(RESULTS / "b4" / "b4_comparison_summary.csv", index=False)
P_ = pd.DataFrame(prow)
for t in ("student", "module"): P_[f"p_{t}_holm"] = holm(P_[f"p_{t}"].values)
P_.round(4).to_csv(RESULTS / "b4" / "b4_paired_comparisons.csv", index=False)
print(M.to_string(index=False)); print(); print(P_.round(3)[["week", "a", "b", "diff", "ci_module_lo", "ci_module_hi", "p_student_holm", "p_module_holm"]].to_string(index=False))
