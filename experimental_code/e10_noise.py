"""E10: readout noise and finite-shot sensitivity of the trained A4-ring circuit (Week 12, 2014J test). SIMULATOR ONLY: no quantum hardware was used.
The circuit is retrained with the stored settings (lr 0.003, 55 epochs, seeds 0-4); the ideal AUC must equal the stored e6_final.csv value (reproducibility check).
Noise is applied to the measured readout only, then fed to the already-trained classifier head (no retraining under noise):
  ideal            exact statevector expectation values
  shots S          ZZ_i estimated from S measurement shots per student (multinomial sampling of the computational-basis distribution), 20 independent draws
  readout flip q   each measured qubit flipped with probability q -> <Z_s Z_b> multiplied by (1-2q)^2 (exact for independent symmetric bit flips)
  depolarising p   global depolarising channel on the final state -> every non-identity Pauli expectation multiplied by (1-p)
Per-gate noise models (noise inside the circuit) were NOT evaluated."""
import csv, numpy as np, jax, jax.numpy as jnp
import train_e6 as E, qsim_topo as T, data_lib
from sklearn.metrics import roc_auc_score
from paths import RESULTS
arm, n, wk = "A4-ring", 4, 12; lr, epochs = 0.003, 55
d = data_lib.load_week(wk); y = d["y"].astype("int32"); tune_m, val_m, full_m, test_m = E.split_masks(d); fi, xi = np.where(full_m)[0], np.where(test_m)[0]
X = E.preprocess(d, full_m); sg = T.signs(n); zzs = np.asarray(sg[0]); rng = np.random.RandomState(10); yt = y[xi]
stored = {int(r["seed"]): float(r["auc"]) for r in csv.DictReader(open(RESULTS / "e6_final.csv")) if r["arm"] == arm and int(r["week"]) == wk}
def head(p, e, xd):
    z = jnp.concatenate([e, xd], 1); h = jax.nn.gelu(z @ p["W1"] + p["b1"]); h = jax.nn.gelu(h @ p["W2"] + p["b2"]); return np.asarray(jax.nn.softmax(h @ p["W3"] + p["b3"])[:, 1])
rows = []
for seed in range(5):
    p, _ = E.train(arm, X, y, fi, lr, seed, sg, epochs=epochs)
    ts = jnp.pi * jnp.tanh(X["sem"][xi] @ p["Ws"] + p["bs"]); tb = jnp.pi * jnp.tanh(X["beh"][xi] @ p["Wb"] + p["bb"])
    psi = T.run_circuit(p, ts, tb, "ring"); zz = np.asarray(T.observables(psi, n, sg)["zz"]); xd = X["demo"][xi]; probs = np.asarray(psi) ** 2
    def auc_for(zz_):
        e = jnp.concatenate([jnp.asarray(zz_.mean(1, keepdims=True)), jnp.asarray(zz_)], 1); return roc_auc_score(yt, head(p, e, xd))
    ideal = auc_for(zz); rows.append(dict(seed=seed, cond="ideal", level=0, draw=0, auc=ideal, stored_auc=stored[seed], match=abs(ideal - stored[seed]) < 1e-4))
    for q in (0.01, 0.02, 0.05, 0.10): rows.append(dict(seed=seed, cond="readout_flip", level=q, draw=0, auc=auc_for(zz * (1 - 2 * q) ** 2)))
    for pd_ in (0.01, 0.05, 0.10, 0.20): rows.append(dict(seed=seed, cond="depolarising", level=pd_, draw=0, auc=auc_for(zz * (1 - pd_))))
    probs64 = probs.astype(np.float64); probs64 /= probs64.sum(axis=1, keepdims=True)          # renormalise in float64 (float32 sums can exceed 1 after casting)
    for S in (10, 100, 1000, 10000):
        for draw in range(20):
            counts = np.stack([rng.multinomial(S, np.minimum(pr, 1.0)) if False else rng.multinomial(S, pr[:-1].tolist() + [max(0.0, 1.0 - float(pr[:-1].sum()))]) for pr in probs64])
            rows.append(dict(seed=seed, cond="shots", level=S, draw=draw, auc=auc_for(counts @ zzs / S)))
    print("seed", seed, "ideal AUC %.4f (stored %.4f)" % (ideal, stored[seed]), flush=True)
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(RESULTS / "e10" / "e10_noise_raw.csv", index=False)
s = df.groupby(["cond", "level"]).auc.agg(["mean", "std", "min"]).reset_index(); s.round(4).to_csv(RESULTS / "e10" / "e10_noise_summary.csv", index=False); print(s.round(4).to_string(index=False))
print("reproducibility: ideal matches stored e6 value for", int(df[df.cond == "ideal"].match.sum()), "of 5 seeds")
