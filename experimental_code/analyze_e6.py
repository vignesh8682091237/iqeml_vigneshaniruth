"""Statistics for E6/E3/E1: five-seed summaries, student- and module-clustered bootstrap, paired comparisons with Holm correction,
per-module AUC, calibration (ECE + reliability diagrams), runtime, and QECS validation. Reads results/preds/*.npy and results/*.csv."""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata, wilcoxon, spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from paths import RESULTS, PREDS
import data_lib
from train_e6 import split_masks
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; WEEKS = (4, 8, 12); rng = np.random.RandomState(2026)
ARMS = [("B1_LogReg", "trees_B1_LogReg"), ("B2_RF", "trees_B2_RF"), ("B3_HistGB", "trees_B3_HistGB"), ("A0_RF", "trees_A0_RF"), ("A1_RF", "trees_A1_RF"), ("A1n", "A1n"), ("A2", "A2"), ("A2s", "A2s"),
        ("A3", "A3"), ("A4-intra", "A4-intra"), ("A4-pair", "A4-pair"), ("A4-ring", "A4-ring"), ("A4-all", "A4-all"), ("A5", "A5")]
PAIRS = [("A4-ring", "A4-intra"), ("A4-pair", "A4-intra"), ("A4-all", "A4-intra"), ("A4-ring", "A4-pair"), ("A4-all", "A4-ring"), ("A4-ring", "A2s"), ("A4-all", "A2s"),
         ("A4-ring", "A2"), ("A5", "A4-ring"), ("A3", "A4-ring"), ("A2s", "A2"), ("A4-ring", "A1_RF")]
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0.
    for rank, i in enumerate(o): run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1., run)
    return adj
def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0.
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
final = pd.concat([pd.read_csv(RESULTS / "e6_final.csv"), pd.read_csv(RESULTS / "trees_final.csv")], ignore_index=True)
main, paired, permod, eces, diagrows = [], [], [], [], []
fig_data = {}
for wk in WEEKS:
    d = data_lib.load_week(wk); _, _, _, tm = split_masks(d); yt = d["y"][tm].astype(int); mod = d["module"][tm]; n = len(yt)
    P = {}
    for name, pre in ARMS:
        fs = sorted(PREDS.glob(f"{pre}_w{wk}_s?.npy"))
        if fs: P[name] = [np.load(f) for f in fs]
    if not P: continue
    sidx = rng.randint(0, n, (R, n)).astype(np.int32); mods = np.unique(mod); rows_of = {m: np.where(mod == m)[0] for m in mods}
    cidx = []
    for _ in range(R):
        pick = rng.choice(mods, len(mods), replace=True); ix = np.concatenate([rows_of[m] for m in pick])
        cidx.append(ix if 0 < yt[ix].sum() < len(ix) else np.concatenate([rows_of[m] for m in mods]))
    boots = {}
    for name, ps in P.items():
        pt = np.array([auc(yt, p) for p in ps]); bs = np.array([np.mean([auc(yt[i], p[i]) for p in ps]) for i in sidx]); bc = np.array([np.mean([auc(yt[i], p[i]) for p in ps]) for i in cidx])
        boots[name] = (pt, bs, bc); f = final[(final.arm == name.replace("_", "_")) | (final.arm == name)]; f = f[f.week == wk]
        main.append(dict(week=wk, arm=name, n_seeds=len(ps), auc=pt.mean(), auc_sd_seeds=pt.std(ddof=1) if len(pt) > 1 else np.nan,
                         ci_student_lo=np.percentile(bs, 2.5), ci_student_hi=np.percentile(bs, 97.5), ci_module_lo=np.percentile(bc, 2.5), ci_module_hi=np.percentile(bc, 97.5),
                         pr_auc=f.pr_auc.mean(), f1=f.f1.mean(), recall=f.recall.mean(), brier=f.brier.mean(), ece=np.mean([ece(yt, p) for p in ps]),
                         secs_train=f.secs_train.mean() if "secs_train" in f else np.nan, params_total=f.params_total.mean() if "params_total" in f else np.nan, params_fusion=f.params_fusion.mean() if "params_fusion" in f else np.nan))
        for m in mods: permod.append(dict(week=wk, arm=name, module=m, n=len(rows_of[m]), at_risk=yt[rows_of[m]].mean(), auc=np.mean([auc(yt[rows_of[m]], p[rows_of[m]]) for p in ps])))
        fig_data[(wk, name)] = ps[0]
    for a, b in PAIRS:
        if a not in boots or b not in boots: continue
        da = boots[a][0] - boots[b][0] if len(boots[a][0]) == len(boots[b][0]) else boots[a][0].mean() - np.repeat(boots[b][0], len(boots[a][0]))
        row = dict(week=wk, a=a, b=b, diff=boots[a][0].mean() - boots[b][0].mean())
        for tag, k in (("student", 1), ("module", 2)):
            dd = boots[a][k] - boots[b][k]; row[f"ci_{tag}_lo"], row[f"ci_{tag}_hi"] = np.percentile(dd, [2.5, 97.5]); row[f"p_{tag}"] = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1. / (R + 1))
        row["d_z_seeds"] = da.mean() / da.std(ddof=1) if len(da) > 1 and da.std(ddof=1) > 0 else np.nan
        try: row["wilcoxon_p_seeds"] = wilcoxon(da).pvalue if len(da) > 1 and np.any(da != 0) else np.nan
        except ValueError: row["wilcoxon_p_seeds"] = np.nan
        paired.append(row)
    for name in ("A3", "A4-intra", "A4-pair", "A4-ring", "A4-all", "A5"):                                             # QECS validation
        for s in range(5):
            f = PREDS / f"{name}_w{wk}_s{s}_diag.npz"
            if not f.exists(): continue
            z = np.load(f); r = {"week": wk, "arm": name, "seed": s}
            for nm in ("qecs", "cqecs"):
                sign = 1. if auc(z["y_tr"], z[nm + "_tr"]) >= 0.5 else -1.; a_or = auc(z["y"], sign * z[nm]); n1 = z["y"].sum(); n0 = len(z["y"]) - n1; Q1, Q2 = a_or / (2 - a_or), 2 * a_or ** 2 / (1 + a_or)
                se = np.sqrt((a_or * (1 - a_or) + (n1 - 1) * (Q1 - a_or ** 2) + (n0 - 1) * (Q2 - a_or ** 2)) / (n1 * n0)); g1, g0 = z[nm][z["y"] == 1], z[nm][z["y"] == 0]
                r.update({f"{nm}_sign_train": sign, f"{nm}_auc_oriented": a_or, f"{nm}_auc_signinv": max(a_or, 1 - a_or), f"{nm}_lcb": a_or - 1.96 * se, f"{nm}_cohen_d_oriented": sign * (g1.mean() - g0.mean()) / np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2),
                          f"{nm}_mean_abs": np.abs(z[nm]).mean(), f"{nm}_mean": z[nm].mean()})
            r.update(mean_sent=z["sent"].mean(), pearson_qecs_sent=pearsonr(z["qecs"], z["sent"])[0] if z["sent"].std() > 1e-9 else np.nan, spearman_qecs_sent=spearmanr(z["qecs"], z["sent"])[0] if z["sent"].std() > 1e-9 else np.nan,
                     pearson_cqecs_sent=pearsonr(z["cqecs"], z["sent"])[0] if z["sent"].std() > 1e-9 else np.nan)
            diagrows.append(r)
M = pd.DataFrame(main); M.round(4).to_csv(RESULTS / "e6_main_results.csv", index=False)
Pr = pd.DataFrame(paired)
for tag in ("student", "module"): Pr[f"p_{tag}_holm"] = holm(Pr[f"p_{tag}"].values)
Pr.round(4).to_csv(RESULTS / "e6_paired_comparisons.csv", index=False); pd.DataFrame(permod).round(4).to_csv(RESULTS / "e6_per_module_auc.csv", index=False)
Q = pd.DataFrame(diagrows); Q.round(4).to_csv(RESULTS / "e6_qecs_per_seed.csv", index=False)
if len(Q):
    g = Q.groupby(["arm", "week"]); S = g.agg(n_seeds=("seed", "count"), qecs_auc_signinv=("qecs_auc_signinv", "mean"), qecs_auc_oriented=("qecs_auc_oriented", "mean"), seeds_lcb_above_half=("qecs_lcb", lambda v: int((v > 0.5).sum())),
        cqecs_auc_oriented=("cqecs_auc_oriented", "mean"), cqecs_seeds_lcb_above_half=("cqecs_lcb", lambda v: int((v > 0.5).sum())), qecs_cohen_d=("qecs_cohen_d_oriented", "mean"), cqecs_cohen_d=("cqecs_cohen_d_oriented", "mean"),
        mean_abs_cqecs=("cqecs_mean_abs", "mean"), mean_sent=("mean_sent", "mean"), corr_qecs_sent=("pearson_qecs_sent", "mean"), corr_cqecs_sent=("pearson_cqecs_sent", "mean"), sign_flip_share=("qecs_sign_train", lambda v: float((v < 0).mean())))
    S.round(4).to_csv(RESULTS / "e6_qecs_validation.csv")
(RESULTS / "figures").mkdir(exist_ok=True)
for wk in WEEKS:
    d = data_lib.load_week(wk); _, _, _, tm = split_masks(d); yt = d["y"][tm]; names = [n for n in ("A1_RF", "A2s", "A4-ring", "A5") if (wk, n) in fig_data]
    if not names: continue
    fig, ax = plt.subplots(figsize=(4.6, 4.4)); ax.plot([0, 1], [0, 1], "k:", lw=1)
    for nm in names:
        p = fig_data[(wk, nm)]; q = np.quantile(p, np.linspace(0, 1, 11)); q[-1] += 1e-9; xs, ys = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            m = (p >= lo) & (p < hi)
            if m.any(): xs.append(p[m].mean()); ys.append(yt[m].mean())
        ax.plot(xs, ys, "o-", ms=3.5, label=nm)
    ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed at-risk rate"); ax.set_title(f"Reliability, Week {wk} (seed 0, 2014J)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULTS / "figures" / f"reliability_week{wk}.png", dpi=170); plt.close(fig)
print(M[["week", "arm", "n_seeds", "auc", "ci_student_lo", "ci_student_hi", "ci_module_lo", "ci_module_hi", "ece", "secs_train"]].round(3).to_string(index=False))
print(Pr[["week", "a", "b", "diff", "ci_student_lo", "ci_student_hi", "p_student_holm", "p_module_holm", "d_z_seeds"]].round(3).to_string(index=False))
