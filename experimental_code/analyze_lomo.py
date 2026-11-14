"""E7 analysis: per-module metrics, seed-level summaries, module-clustered bootstrap CIs and Holm-corrected paired comparisons.
Reads results/lomo/lomo_fold_metrics.csv and the saved per-fold predictions. Writes results/lomo/*.csv."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from paths import RESULTS, COMPACT
import data_lib
OUT = RESULTS / "lomo"; R = 1000; rng = np.random.RandomState(7)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.
    for rank, i in enumerate(o): run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1., run)
    return adj

fm = pd.read_csv(OUT / "lomo_fold_metrics.csv")
# 1. per-module summary: mean and SD over seeds, per arm, week, module
per_mod = fm.groupby(["arm", "week", "module"]).agg(n_seeds=("seed", "count"), n_test=("n_test", "first"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
           pr_auc=("pr_auc", "mean"), f1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"), brier=("brier", "mean")).reset_index()
per_mod.round(4).to_csv(OUT / "lomo_per_module_summary.csv", index=False)

# 2. overall LOMO score per arm and week: mean over modules, module-clustered bootstrap CI (resample modules)
def arm_week_scores(arm, wk):
    g = fm[(fm.arm == arm) & (fm.week == wk)].groupby(["module", "seed"]).auc.mean().unstack("seed")
    return g
rows = []; mods_all = sorted(fm.module.unique())
for arm in ("A1", "A2", "A4-ring"):
    for wk in (4, 8, 12):
        g = arm_week_scores(arm, wk)                       # module x seed
        per_seed_mean = g.mean(axis=0).values                   # average over modules, one value per seed
        boots = [g.iloc[rng.choice(len(g), len(g), replace=True)].mean(axis=0).values for _ in range(R)]
        boots = np.array(boots).mean(1)                    # bootstrap distribution of the seed-averaged mean
        rows.append(dict(arm=arm, week=wk, n_modules=len(g), lomo_auc_mean=per_seed_mean.mean(),
                         seed_sd=per_seed_mean.std(ddof=1) if len(per_seed_mean) > 1 else np.nan,
                         ci_module_lo=np.percentile(boots, 2.5), ci_module_hi=np.percentile(boots, 97.5)))
overall = pd.DataFrame(rows); overall.round(4).to_csv(OUT / "lomo_overall_summary.csv", index=False)

# 3. paired comparisons across modules (same module, same week): Holm over all tested pairs
pairs = [("A4-ring", "A1"), ("A4-ring", "A2"), ("A2", "A1")]
prow = []
for a, b in pairs:
    for wk in (4, 8, 12):
        ga, gb = arm_week_scores(a, wk), arm_week_scores(b, wk)
        mod_diff = (ga.mean(axis=1) - gb.mean(axis=1)).reindex(ga.index)         # one difference per module, seed-averaged
        boots = np.array([mod_diff.iloc[rng.choice(len(mod_diff), len(mod_diff), replace=True)].mean() for _ in range(R)])
        p = max(2 * min((boots <= 0).mean(), (boots >= 0).mean()), 1. / (R + 1))
        prow.append(dict(a=a, b=b, week=wk, diff=mod_diff.mean(), ci_module_lo=np.percentile(boots, 2.5), ci_module_hi=np.percentile(boots, 97.5),
                         modules_a_better=int((mod_diff > 0).sum()), n_modules=len(mod_diff), p_boot=p))
P = pd.DataFrame(prow); P["p_holm"] = holm(P.p_boot.values); P.round(4).to_csv(OUT / "lomo_paired_comparisons.csv", index=False)
print(overall.round(3).to_string(index=False)); print(); print(P.round(3).to_string(index=False))
