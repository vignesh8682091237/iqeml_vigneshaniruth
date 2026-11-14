"""E6 + E3 + E1: presentation-held-out validation, topology ablation and parameter-matched classical control (JAX, CPU).
Stage 'tune' : train on 2013B+2013J, early-stop/select on 2014B (validation only). 2014J is never touched.
Stage 'refit': refit on 2013B+2013J+2014B with the selected learning rate and epoch count, evaluate ONCE on 2014J (5 seeds)."""
import argparse, os, time, csv, json, numpy as np, jax, jax.numpy as jnp
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, recall_score
from paths import RESULTS, PREDS, ROOT
import data_lib, qsim_topo as T
jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))

N_Q, BS, PAT, MAXEP, TAU = 4, 512, 6, 60, 0.07
TUNE_TRAIN, VAL, TEST = ("2013B", "2013J"), ("2014B",), ("2014J",)
ARMS = {"A1n": ("mlp", None), "A2": ("a2", None), "A2s": ("a2s", None), "A3": ("amp", "ring"),
        "A4-intra": ("qecs", "intra"), "A4-pair": ("qecs", "pair"), "A4-ring": ("qecs", "ring"), "A4-all": ("qecs", "all"), "A5": ("full", "ring"),
        "A2-w": ("a2", None), "A2s-w": ("a2s", None), "A4-ring-w": ("qecs", "ring")}      # *-w: same models, tuned with a wider learning-rate grid (and a smaller init for A2s-w)
TARGET = "T2"; TUNE_CSV, FINAL_CSV = RESULTS / "e6_tuning.csv", RESULTS / "e6_final.csv"; PFX = ""

def set_target(t):
    global TARGET, TUNE_CSV, FINAL_CSV, PFX
    TARGET = t
    if t != "T2": TUNE_CSV, FINAL_CSV, PFX = RESULTS / f"e6_tuning_{t}.csv", RESULTS / f"e6_final_{t}.csv", f"{t}_"

def apply_target(d, wk):
    """T1: Withdrawn vs all others. T3: Withdrawn vs Pass only (Fail and Distinction rows removed everywhere). T2 (default) is unchanged."""
    if TARGET == "T2": return d
    fr = np.load(data_lib.COMPACT / f"week{wk}_final.npz", allow_pickle=True)["final_result"]; assert len(fr) == len(d["y"]) and np.array_equal((np.isin(fr, ["Withdrawn", "Fail"])).astype("int32"), d["y"])
    keep = np.ones(len(fr), bool) if TARGET == "T1" else np.isin(fr, ["Withdrawn", "Pass"]); y = (fr == "Withdrawn").astype("int32")
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(fr) and not k.endswith("_cols") else v) for k, v in d.items()}; out["y"] = y[keep]; return out

def slog(x): return np.sign(x) * np.log1p(np.abs(x))
def split_masks(d):
    p = d["pres"]; tune, val, test = np.isin(p, TUNE_TRAIN), np.isin(p, VAL), np.isin(p, TEST); full = tune | val
    assert not (full & test).any() and not (tune & val).any() and (tune | val | test).all()
    return tune, val, full, test
def preprocess(d, fit_mask):
    out = {}
    for k in ("beh", "sem", "demo"):
        x = slog(d[k]); mu, sd = x[fit_mask].mean(0), x[fit_mask].std(0) + 1e-6; out[k] = ((x - mu) / sd).astype("float32")
    return out

def init(arm, key, ds, db, dd, n=N_Q):
    kind, topo = ARMS[arm]; ks = iter(jax.random.split(key, 20)); nrm = lambda shape, sc: jax.random.normal(next(ks), shape) * sc; p = {}
    if kind == "mlp": din = db + ds + dd
    else:
        p["Ws"], p["bs"] = nrm((ds, n), ds ** -.5), jnp.zeros(n); p["Wb"], p["bb"] = nrm((db, n), db ** -.5), jnp.zeros(n)
        edim = {"a2": n + 2, "a2s": n + 1, "amp": n + 2, "qecs": n + 1, "full": n + 2}[kind]; din = edim + dd
        if kind == "a2": p["Wq"], p["bq"] = nrm((8 * n, n + 2), (8 * n) ** -.5), jnp.zeros(n + 2)
        if kind == "a2s":
            for nm in ("fa", "fb", "fc", "fd"): p[nm] = nrm((n,), 0.1 if arm == "A2s-w" else 1.0)               # 4n = 16 fusion parameters (matched to the ring circuit)
        if topo is not None:
            for nm, sh in T.param_shapes(n, topo).items():
                p[nm] = nrm(sh, 0.1) if nm == "phi" else jax.random.uniform(next(ks), sh, minval=-1., maxval=1.)
        if kind == "amp": p["Wa"], p["ba"] = nrm((2 ** (2 * n), n + 2), 1.0), jnp.zeros(n + 2)
    p["W1"], p["b1"] = nrm((din, 64), din ** -.5), jnp.zeros(64); p["W2"], p["b2"] = nrm((64, 32), 64 ** -.5), jnp.zeros(32)
    p["W3"], p["b3"] = nrm((32, 2), 32 ** -.5), jnp.zeros(2)
    return p

def param_report(arm, p):
    sz = lambda keys: int(sum(int(np.prod(p[k].shape)) for k in keys if k in p))
    fusion = sz(["Wq", "bq", "fa", "fb", "fc", "fd", "lam", "phi", "mu", "Wa", "ba"])
    return dict(params_total=sz(list(p)), params_fusion=fusion, params_circuit=sz(["lam", "phi", "mu"]), params_proj=sz(["Ws", "bs", "Wb", "bb"]), params_head=sz(["W1", "b1", "W2", "b2", "W3", "b3"]))

def forward(arm, p, xs, xb, xd, sg, n=N_Q, want_obs=False):
    kind, topo = ARMS[arm]; ob = None
    if kind == "mlp": z = jnp.concatenate([xb, xs, xd], 1); e = None
    else:
        ts = jnp.pi * jnp.tanh(xs @ p["Ws"] + p["bs"]); tb = jnp.pi * jnp.tanh(xb @ p["Wb"] + p["bb"])
        if kind == "a2":
            cs, ss, cb, sb = jnp.cos(ts), jnp.sin(ts), jnp.cos(tb), jnp.sin(tb)
            e = jnp.tanh(jnp.concatenate([cs, ss, cb, sb, cs * cb, ss * sb, cs * sb, ss * cb], 1) @ p["Wq"] + p["bq"])
        elif kind == "a2s":
            cs, ss, cb, sb = jnp.cos(ts), jnp.sin(ts), jnp.cos(tb), jnp.sin(tb)
            ei = jnp.tanh(p["fa"] * cs * cb + p["fb"] * ss * sb + p["fc"] * cs * sb + p["fd"] * ss * cb); e = jnp.concatenate([ei.mean(1, keepdims=True), ei], 1)
        else:
            psi = T.run_circuit(p, ts, tb, topo); ob = T.observables(psi, n, sg)
            if kind == "amp": e = psi @ p["Wa"] + p["ba"]
            elif kind == "qecs": e = jnp.concatenate([ob["qecs"][:, None], ob["zz"]], 1)
            else: e = jnp.concatenate([ob["qecs"][:, None], jax.lax.stop_gradient(ob["sent"])[:, None], ob["zz"]], 1)
        z = jnp.concatenate([e, xd], 1)
    h = jax.nn.gelu(z @ p["W1"] + p["b1"]); h = jax.nn.gelu(h @ p["W2"] + p["b2"]); lg = h @ p["W3"] + p["b3"]
    return (lg, e, ob) if want_obs else (lg, e)

def supcon(e, y):
    en = e / (jnp.linalg.norm(e, axis=1, keepdims=True) + 1e-8); s = en @ en.T / TAU; eye = jnp.eye(len(y), dtype=bool)
    s = jnp.where(eye, -1e9, s); s = s - jax.lax.stop_gradient(s.max(1, keepdims=True)); ex = jnp.exp(s)
    pos = (y[:, None] == y[None, :]) & ~eye; num, den, ok = (ex * pos).sum(1), ex.sum(1), pos.sum(1) > 0
    return -(jnp.where(ok, jnp.log((num + 1e-8) / (den + 1e-8)), 0.).sum() / ok.sum())

def loss_fn(p, arm, xs, xb, xd, y, sg):
    lg, e = forward(arm, p, xs, xb, xd, sg); lp = jax.nn.log_softmax(lg)
    ce = -jnp.take_along_axis(lp, y[:, None], 1)[:, 0]; pt = jnp.exp(-ce); focal = (0.25 * (1 - pt) ** 2 * ce).mean()
    smooth = -((jax.nn.one_hot(y, 2) * 0.95 + 0.025) * lp).sum(1).mean()
    return 0.6 * focal + 0.25 * smooth + 0.15 * supcon(e, y) if ARMS[arm][0] == "full" else 0.706 * focal + 0.294 * smooth   # fixed weights (not tuned)

@partial(jax.jit, static_argnames=("arm",))
def step(p, m, v, t, lr, xs, xb, xd, y, sg, arm):
    l, g = jax.value_and_grad(loss_fn)(p, arm, xs, xb, xd, y, sg)
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g); v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
    p = jax.tree_util.tree_map(lambda w, a, b: w - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), p, m, v)
    return p, m, v, l

@partial(jax.jit, static_argnames=("arm", "obs"))
def fwd_eval(p, xs, xb, xd, sg, arm, obs):
    lg, e, ob = forward(arm, p, xs, xb, xd, sg, want_obs=True); out = {"prob": jax.nn.softmax(lg)[:, 1]}
    if obs and ob is not None: out.update(qecs=ob["qecs"], cqecs=ob["cqecs"], sent=ob["sent"])
    return out

def predict(arm, p, X, idx, sg, obs=False, B=2048):
    res = {}
    for i in range(0, len(idx), B):
        j = idx[i:i + B]; jj = np.concatenate([j, np.repeat(j[-1:], B - len(j))]); o = fwd_eval(p, X["sem"][jj], X["beh"][jj], X["demo"][jj], sg, arm, obs)
        for k, a in o.items(): res.setdefault(k, []).append(np.asarray(a)[:len(j)])
    return {k: np.concatenate(v) for k, v in res.items()}

def train(arm, X, y, fit_idx, lr, seed, sg, epochs=None, val_idx=None):
    ds, db, dd = X["sem"].shape[1], X["beh"].shape[1], X["demo"].shape[1]; p = init(arm, jax.random.PRNGKey(seed), ds, db, dd)
    m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p); rng = np.random.RandomState(seed)
    j0 = fit_idx[:BS]; step(p, m, v, 1, lr, X["sem"][j0], X["beh"][j0], X["demo"][j0], y[j0], sg, arm)          # warm-up (compile), result discarded -> timing excludes compilation
    t0 = time.time(); best, best_p, best_ep, bad, t = -1., p, 0, 0, 0
    for ep in range(MAXEP if epochs is None else epochs):
        perm = rng.permutation(fit_idx)
        for b in range(0, len(perm) - BS + 1, BS):
            j = perm[b:b + BS]; t += 1; p, m, v, l = step(p, m, v, t, lr, X["sem"][j], X["beh"][j], X["demo"][j], y[j], sg, arm)
        if val_idx is not None:
            va = roc_auc_score(y[val_idx], predict(arm, p, X, val_idx, sg)["prob"])
            if va > best: best, best_p, best_ep, bad = va, p, ep, 0
            else:
                bad += 1
                if bad >= PAT: break
    return (best_p if val_idx is not None else p), dict(best_val_auc=best, best_epoch=best_ep, secs_train=time.time() - t0, params=param_report(arm, p))

def append_row(path, row, fields):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); (new and w.writeheader()); w.writerow(row)

def read_rows(path):
    if not path.exists(): return []
    with open(path, newline="") as f: return list(csv.DictReader(f))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["tune", "refit"]); ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--weeks", default="4,8,12"); ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--lrs", default="0.001,0.003,0.01"); ap.add_argument("--target", default="T2", choices=["T1", "T2", "T3"])
    a = ap.parse_args(); set_target(a.target); sg = T.signs(N_Q)
    for wk in [int(w) for w in a.weeks.split(",")]:
        d = apply_target(data_lib.load_week(wk), wk); y = d["y"].astype("int32"); tune_m, val_m, full_m, test_m = split_masks(d); ti, vi, fi, xi = (np.where(m)[0] for m in (tune_m, val_m, full_m, test_m))
        for arm in a.arms.split(","):
            if a.stage == "tune":
                X = preprocess(d, tune_m); assert not test_m[ti].any() and not test_m[vi].any()          # test presentation never enters tuning
                for lr in [float(x) for x in a.lrs.split(",")]:
                    if any(r["arm"] == arm and int(r["week"]) == wk and float(r["lr"]) == lr for r in read_rows(TUNE_CSV)): continue
                    _, info = train(arm, X, y, ti, lr, 0, sg, val_idx=vi)
                    append_row(TUNE_CSV, dict(arm=arm, week=wk, lr=lr, val_auc=round(info["best_val_auc"], 5), best_epoch=info["best_epoch"], secs_train=round(info["secs_train"], 1), **info["params"]),
                               ["arm", "week", "lr", "val_auc", "best_epoch", "secs_train", "params_total", "params_fusion", "params_circuit", "params_proj", "params_head"])
                    print("tune", arm, wk, lr, "val AUC %.4f ep %d" % (info["best_val_auc"], info["best_epoch"]), flush=True)
            else:
                rows = [r for r in read_rows(TUNE_CSV) if r["arm"] == arm and int(r["week"]) == wk]; assert rows, f"run tuning first for {arm} week {wk}"
                best = max(rows, key=lambda r: float(r["val_auc"])); lr, epochs = float(best["lr"]), int(best["best_epoch"]) + 1
                X = preprocess(d, full_m); rs = np.random.RandomState(99).choice(fi, 3000, replace=False)
                for seed in [int(s) for s in a.seeds.split(",")]:
                    pf = PREDS / f"{PFX}{arm}_w{wk}_s{seed}.npy"
                    if pf.exists(): continue
                    p, info = train(arm, X, y, fi, lr, seed, sg, epochs=epochs)
                    o = predict(arm, p, X, xi, sg, obs=True); pr = o["prob"]; yt = y[xi]; np.save(pf, pr.astype("float32"))     # the only place 2014J is evaluated
                    if "qecs" in o:
                        otr = predict(arm, p, X, rs, sg, obs=True)
                        np.savez_compressed(PREDS / f"{PFX}{arm}_w{wk}_s{seed}_diag.npz", qecs=o["qecs"], cqecs=o["cqecs"], sent=o["sent"], y=yt, qecs_tr=otr["qecs"], cqecs_tr=otr["cqecs"], y_tr=y[rs])
                    row = dict(arm=arm, week=wk, seed=seed, lr=lr, epochs=epochs, val_auc_tuning=best["val_auc"], auc=roc_auc_score(yt, pr), pr_auc=average_precision_score(yt, pr), f1=f1_score(yt, pr > 0.5),
                               recall=recall_score(yt, pr > 0.5), brier=brier_score_loss(yt, pr), secs_train=round(info["secs_train"], 1), **info["params"])
                    append_row(FINAL_CSV, row, ["arm", "week", "seed", "lr", "epochs", "val_auc_tuning", "auc", "pr_auc", "f1", "recall", "brier", "secs_train", "params_total", "params_fusion", "params_circuit", "params_proj", "params_head"])
                    print("refit", arm, wk, seed, "test AUC %.4f (lr %g, %d epochs)" % (row["auc"], lr, epochs), "%.0fs" % info["secs_train"], flush=True)
