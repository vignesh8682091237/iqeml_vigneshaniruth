"""E8 - B4: ADAPTED implementation of HQIAT-ML (Bichi & Shekhawat, COMPUTATIA 2026, hqiat.pdf read from the project Drive).

THIS IS NOT AN EXACT REPRODUCTION. Source-specified settings are used as published; the following were ambiguous or absent and
were set by the choices documented below (these are the adaptations, reported in V13_REPRODUCTION_GUIDE.md):
  A1  Eq.(1): cos(theta)*psi + sin(phi)*psi taken literally (element-wise, learnable theta, phi). Note this is a scalar gain per
      dimension, not a phase rotation.
  A2  Eq.(2): "psi_phase . E" interpreted as psi_phase @ E (matrix product, E in R^{q x q}); normalize = L2 normalisation over q.
  A3  Eq.(3): x_out = x + 0.3 * W_m(psi_ent), W_m in R^{q x d} (residual keeps dimension d).
  A4  Attention needs a token sequence. The tabular vector is projected to L = 4 tokens of width C = 32 (one per feature group
      conceptually; here a learned projection). The paper's attention/gated-residual equations (4)-(6) are then applied as written
      (h = 4 heads, gamma init 0.1, MLP expansion 4, GELU, dropout 0.15).
  A5  Head-importance softmax: learnable per-head logits applied to the head outputs (the paper's O_i formula is ambiguous).
  A6  Multi-scale temporal convolutions (kernels 1,3,5) OMITTED: with a length-4 token sequence and no temporal order in the data,
      they have no defined input; attention pooling replaced by mean pooling over tokens.
  A7  Ensemble: three heads, each MLP d_in->256->128->1 with GELU and dropout 0.075; prediction sigma(head); combined with
      softmax weights (Eq. 7).
  A8  Mixup with probability 0.5 and Beta(0.2,0.2) on features and (soft) labels; focal term evaluated on soft labels.
  A9  Robust scaling (median/IQR fitted on training rows) applied to the same raw features used by A1 (all features, no signed log).
  A10 Early stopping: patience 30 on validation AUC (2014B), max 200 epochs, as published. Refit uses the selected epoch count
      on all training presentations (same protocol as the main experiments).
  A12 Input transform (CHANGED after a diagnosed failure): the paper's robust (median/IQR) scaling produces inputs of order 1e6 for
      sparse count features whose training IQR is near zero, which drove the network to chance level (test AUC 0.540, diagnosed).
      B4 therefore uses the same signed-log + z-score transform as the A-arms, clipped to the training-row range.
  A13 Input set: B4 receives the same behavioural + semantic + demographic features as A1/A2/A4 (the paper uses engineered tabular
      features only, without the activity-type features). This is a fairness choice for this study, not a paper feature.
  A11 No hyperparameter tuning: all values are the published ones. The comparison with A1/A2/A4 (which were tuned on the learning
      rate or depth grids) is therefore asymmetric, and this is stated wherever B4 is reported.
Not used: the counterfactual module (not part of the classification claim) and the SHAP comparison.
"""
import json, time, numpy as np, jax, jax.numpy as jnp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, recall_score, precision_score
from paths import RESULTS
import data_lib
import train_e6 as E

Q, L, C, H, ALPHA, EPS = 16, 4, 32, 4, 0.3, 0.05
LAM_F, LAM_S, FOCAL_A, FOCAL_G = 0.7, 0.3, 0.25, 2.0
LR0, WD, T0, TMULT, ETA_MIN, BS, PAT, MAXEP, CLIP = 5e-4, 0.01, 20, 2, 1e-6, 64, 30, 200, 1.0

def robust(X_tr):
    med = np.median(X_tr, 0); iqr = np.percentile(X_tr, 75, 0) - np.percentile(X_tr, 25, 0); return med, np.maximum(iqr, 1e-6)

def init(key, d):
    ks = iter(jax.random.split(key, 64)); n = lambda s, sc: jax.random.normal(next(ks), s) * sc; p = {}
    p["Wphi"] = n((d, Q), d ** -.5); p["theta"] = jnp.zeros(Q); p["phi"] = jnp.zeros(Q)
    p["E"] = jnp.eye(Q) + n((Q, Q), 0.05); p["Wm"] = n((Q, d), Q ** -.5)
    p["Wtok"] = n((d, L * C), d ** -.5); p["bt"] = jnp.zeros(L * C)
    p["Wqkv"] = n((C, 3 * C), C ** -.5); p["Wo"] = n((C, C), C ** -.5); p["hw"] = jnp.zeros(H)
    p["g1"] = jnp.full(C, 0.1); p["g2"] = jnp.full(C, 0.1); p["Wg"] = n((C, C), C ** -.5)
    p["W1"] = n((C, 4 * C), C ** -.5); p["W2"] = n((4 * C, C), (4 * C) ** -.5)
    p["ln1"] = (jnp.ones(C), jnp.zeros(C)); p["ln2"] = (jnp.ones(C), jnp.zeros(C))
    p["hw_ens"] = jnp.zeros(3)
    for k in range(3):
        p[f"h{k}a"] = n((C, 256), C ** -.5); p[f"h{k}b"] = jnp.zeros(256); p[f"h{k}c"] = n((256, 128), 256 ** -.5); p[f"h{k}d"] = jnp.zeros(128)
        p[f"h{k}e"] = n((128, 1), 128 ** -.5); p[f"h{k}f"] = jnp.zeros(1)
    return p

def ln(x, g, b): mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x - mu) / jnp.sqrt(v + 1e-5) * g + b

def forward(p, x, key=None, train=False):
    # A1-A3 quantum-inspired feature transformation (Eqs 1-3, adapted)
    psi = jnp.tanh(x @ p["Wphi"]); psi_ph = (jnp.cos(p["theta"]) + jnp.sin(p["phi"])) * psi
    psi_ent = psi_ph @ p["E"]; psi_ent = psi_ent / (jnp.linalg.norm(psi_ent, axis=1, keepdims=True) + 1e-8)
    xo = x + ALPHA * (psi_ent @ p["Wm"])
    # A4 tokens + adaptive multi-head attention (Eqs 4-6)
    tok = (xo @ p["Wtok"] + p["bt"]).reshape(-1, L, C)
    h1 = ln(tok, *p["ln1"]); qkv = h1 @ p["Wqkv"]; q, k, v = jnp.split(qkv, 3, -1)
    dk = C // H; sp = lambda t: t.reshape(t.shape[0], L, H, dk).transpose(0, 2, 1, 3)
    q, k, v = sp(q), sp(k), sp(v); att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk), -1) @ v
    hw = jax.nn.softmax(p["hw"])                                                           # A5 head importance
    att = (att * hw[None, :, None, None]).transpose(0, 2, 1, 3).reshape(-1, L, C) @ p["Wo"]
    if train: att = att * jax.random.bernoulli(key, 0.85, att.shape) / 0.85
    t1 = tok + p["g1"] * att
    h2 = ln(t1, *p["ln2"]); mlp = jax.nn.gelu(h2 @ p["W1"]) @ p["W2"]
    gate = jax.nn.sigmoid(t1 @ p["Wg"]); t2 = t1 + p["g2"] * gate * mlp
    pooled = t2.mean(1)                                                                    # A6 mean pooling
    # A7 ensemble of three heads (Eq. 7)
    w = jax.nn.softmax(p["hw_ens"]); out = 0.
    for kk in range(3):
        z = jax.nn.gelu(ln(pooled @ p[f"h{kk}a"] + p[f"h{kk}b"], jnp.ones(256), jnp.zeros(256)) if False else pooled @ p[f"h{kk}a"] + p[f"h{kk}b"])
        z = jax.nn.gelu(z @ p[f"h{kk}c"] + p[f"h{kk}d"]); out = out + w[kk] * jax.nn.sigmoid(z @ p[f"h{kk}e"] + p[f"h{kk}f"])[:, 0]
    return jnp.clip(out, 1e-6, 1 - 1e-6)

def loss(p, x, y, key):
    P = forward(p, x, key, train=True)
    pt = y * P + (1 - y) * (1 - P); focal = -FOCAL_A * (1 - pt) ** FOCAL_G * jnp.log(pt)          # A8 soft-label focal
    ys = y * (1 - EPS) + 0.5 * EPS; bce = -(ys * jnp.log(P) + (1 - ys) * jnp.log(1 - P))
    return (LAM_F * focal + LAM_S * bce).mean()

grad_fn = jax.jit(jax.value_and_grad(loss))
pred_fn = jax.jit(lambda p, x: forward(p, x))

def lr_at(ep):   # CosineAnnealingWarmRestarts(T0=20, Tmult=2, eta_min=1e-6), stepped per epoch
    T_i, T_cur = T0, ep
    while T_cur >= T_i: T_cur -= T_i; T_i *= TMULT
    return ETA_MIN + (LR0 - ETA_MIN) * (1 + np.cos(np.pi * T_cur / T_i)) / 2

def adamw_step(p, m, v, g, t, lr):
    gn = jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(g))); s = jnp.minimum(1., CLIP / (gn + 1e-12))
    g = jax.tree_util.tree_map(lambda a: a * s, g)
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g); v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
    p = jax.tree_util.tree_map(lambda w, a, b: w - lr * ((a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8) + WD * w), p, m, v)
    return p, m, v

def run(Xtr, ytr, Xva, yva, seed, epochs=None):
    rng = np.random.RandomState(seed); key = jax.random.PRNGKey(seed); p = init(key, Xtr.shape[1])
    m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p); t = 0
    best, best_p, best_ep, bad = -1., p, 0, 0; n_ep = MAXEP if epochs is None else epochs
    for ep in range(n_ep):
        lr = lr_at(ep); perm = rng.permutation(len(ytr))
        for b0 in range(0, len(perm), BS):
            j = perm[b0:b0 + BS]; xb, yb = Xtr[j], ytr[j].astype(np.float32)
            if rng.rand() < 0.5:                                                     # A8 mixup
                lam = rng.beta(0.2, 0.2); pm = rng.permutation(len(j))
                xb = lam * xb + (1 - lam) * xb[pm]; yb = lam * yb + (1 - lam) * yb[pm]
            key, sk = jax.random.split(key); _, g = grad_fn(p, jnp.asarray(xb), jnp.asarray(yb), sk)
            t += 1; p, m, v = adamw_step(p, m, v, g, t, lr)
        if Xva is not None:
            va = roc_auc_score(yva, np.asarray(pred_fn(p, jnp.asarray(Xva))))
            if va > best: best, best_p, best_ep, bad = va, p, ep, 0
            else:
                bad += 1
                if bad >= PAT: break
    return (best_p if Xva is not None else p), best, best_ep

if __name__ == "__main__":
    import sys, csv
    OUT = RESULTS / "b4"; OUT.mkdir(parents=True, exist_ok=True); weeks = [int(w) for w in (sys.argv[1] if len(sys.argv) > 1 else "12").split(",")]
    seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3,4").split(",")]
    for wk in weeks:
        d = data_lib.load_week(wk); X = np.hstack([d["beh"], d["sem"], d["demo"]]).astype("float32"); y = d["y"].astype("float32"); pres = d["pres"]
        tr = np.isin(pres, ["2013B", "2013J"]); va = np.isin(pres, ["2014B"]); full = tr | va; te = pres == "2014J"
        assert not (full & te).any()
        Xz = E.preprocess(d, tr); X = np.hstack([Xz["beh"], Xz["sem"], Xz["demo"]]).astype("float32"); Xs = np.clip(X, X[tr].min(0), X[tr].max(0))   # A12/A13 (fitted on training rows)
        for seed in seeds:
            f = OUT / f"b4_w{wk}_s{seed}.npy"
            if f.exists(): continue
            t0 = time.time(); _, best, best_ep = run(Xs[tr], y[tr], Xs[va], y[va], seed)                     # tuning-stage early stopping on 2014B
            Xz2 = E.preprocess(d, full); Xf = np.clip(np.hstack([Xz2["beh"], Xz2["sem"], Xz2["demo"]]).astype("float32"), *(lambda A: (A[full].min(0), A[full].max(0)))(np.hstack([Xz2["beh"], Xz2["sem"], Xz2["demo"]])))   # refit: same transform on all training presentations
            p, _, _ = run(Xf[full], y[full], None, None, seed, epochs=best_ep + 1)                            # refit, fixed epoch count
            pr = np.asarray(pred_fn(p, jnp.asarray(Xf[te]))); yt = y[te].astype(int); np.save(f, pr.astype("float32"))
            pred = (pr > 0.5).astype(int)
            row = dict(week=wk, seed=seed, val_auc_2014B=round(best, 5), best_epoch=best_ep, auc=roc_auc_score(yt, pr), pr_auc=average_precision_score(yt, pr),
                       f1=f1_score(yt, pred), precision=precision_score(yt, pred, zero_division=0), recall=recall_score(yt, pred), brier=brier_score_loss(yt, pr), secs=round(time.time() - t0, 1))
            new = not (OUT / "b4_results.csv").exists()
            with open(OUT / "b4_results.csv", "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys())); (new and w.writeheader()); w.writerow(row)
            print(json.dumps(row), flush=True)
