"""E6 for the classical baselines: grid search on 2014B (train 2013B+2013J), refit on all training presentations, evaluate once on 2014J."""
import csv, itertools, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, recall_score
from paths import RESULTS, PREDS
import data_lib
import train_e6
from train_e6 import split_masks, append_row, read_rows
ap = argparse.ArgumentParser(); ap.add_argument("--target", default="T2"); ap.add_argument("--sets", default=""); A = ap.parse_args(); train_e6.set_target(A.target)
PFX = "" if A.target == "T2" else A.target + "_"; SFX = "" if A.target == "T2" else "_" + A.target
TUNE, FINAL = RESULTS / f"trees_tuning{SFX}.csv", RESULTS / f"trees_final{SFX}.csv"
GRIDS = {"lr": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
         "rf": [dict(max_depth=md, min_samples_leaf=ml) for md, ml in itertools.product((6, 12, 20), (1, 10))],
         "gb": [dict(learning_rate=lr, max_depth=md) for lr, md in itertools.product((0.03, 0.1), (3, 6))]}
def make(kind, hp, seed):
    if kind == "lr": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, **hp))
    if kind == "rf": return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed, **hp)
    return HistGradientBoostingClassifier(max_iter=300, random_state=seed, **hp)
for wk in (12, 8, 4):
    d = train_e6.apply_target(data_lib.load_week(wk), wk); y = d["y"]; tm, vm, fm, xm = split_masks(d); B, S, D = d["beh"], d["sem"], d["demo"]
    sets = {"B1_LogReg": (np.hstack([B, D]), "lr"), "B2_RF": (np.hstack([B, D]), "rf"), "B3_HistGB": (np.hstack([B, D]), "gb"), "A0_RF": (B, "rf"), "A1_RF": (np.hstack([B, S, D]), "rf")}
    for name, (X, kind) in sets.items():
        if A.sets and name not in A.sets.split(","): continue
        done = [r for r in read_rows(TUNE) if r["arm"] == name and int(r["week"]) == wk]
        if len(done) < len(GRIDS[kind]):
            for hp in GRIDS[kind]:
                if any(r["hp"] == str(hp) for r in done): continue
                m = make(kind, hp, 0).fit(X[tm], y[tm]); va = roc_auc_score(y[vm], m.predict_proba(X[vm])[:, 1])
                append_row(TUNE, dict(arm=name, week=wk, hp=str(hp), val_auc=round(va, 5)), ["arm", "week", "hp", "val_auc"]); print("tune", name, wk, hp, "val %.4f" % va, flush=True)
        rows = [r for r in read_rows(TUNE) if r["arm"] == name and int(r["week"]) == wk]; best = max(rows, key=lambda r: float(r["val_auc"])); hp = eval(best["hp"])
        for seed in (range(1) if kind == "lr" else range(5)):
            pf = PREDS / f"{PFX}trees_{name}_w{wk}_s{seed}.npy"
            if pf.exists(): continue
            m = make(kind, hp, seed).fit(X[fm], y[fm]); p = m.predict_proba(X[xm])[:, 1]; yt = y[xm]; np.save(pf, p.astype("float32"))
            append_row(FINAL, dict(arm=name, week=wk, seed=seed, hp=str(hp), val_auc_tuning=best["val_auc"], auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p), f1=f1_score(yt, p > 0.5),
                                   recall=recall_score(yt, p > 0.5), brier=brier_score_loss(yt, p)), ["arm", "week", "seed", "hp", "val_auc_tuning", "auc", "pr_auc", "f1", "recall", "brier"])
        print("final", name, wk, hp, flush=True)
