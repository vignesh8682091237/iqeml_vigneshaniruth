"""Batched real-amplitude statevector simulator with selectable cross-modal topology (JAX, CPU).
Wires 0..n-1 = semantic register s, wires n..2n-1 = behaviour register b.  Primitives come from the unchanged original simulator (qsim_legacy).
Topologies (all share: RY angle encoding, CNOT ring inside each register, trainable RY(phi) on all 2n qubits):
  intra : no cross-modal gate at all (product state across registers)            -> trainable circuit params: phi
  pair  : CRY(lam_i) s_i -> b_i                                                  -> lam(n) + phi
  ring  : CRY(lam_i) s_i -> b_i, then CRY(mu_i) s_i -> b_(i+1)   (original QEMF-X) -> lam(n) + phi + mu(n)
  all   : CRY(lam_ij) s_i -> b_j for all i,j, then CRY(mu_ij) for all i,j        -> lam(n^2) + phi + mu(n^2)"""
import numpy as np, jax.numpy as jnp
import qsim_legacy as L
TOPOS = ("intra", "pair", "ring", "all")

def param_shapes(n, topo):
    return {"lam": {"intra": (0,), "pair": (n,), "ring": (n,), "all": (n, n)}[topo], "phi": (2 * n,),
            "mu": {"intra": (0,), "pair": (0,), "ring": (n,), "all": (n, n)}[topo]}

def gate_list(n, topo):
    """Ordered list of (kind, param_name, param_index, control, target)."""
    g = [("CNOT", None, None, i, (i + 1) % n) for i in range(n)] + [("CNOT", None, None, n + i, n + (i + 1) % n) for i in range(n)]
    if topo in ("pair", "ring"): g += [("CRY", "lam", i, i, n + i) for i in range(n)]
    if topo == "all": g += [("CRY", "lam", (i, j), i, n + j) for i in range(n) for j in range(n)]
    g += [("RY", "phi", q, None, q) for q in range(2 * n)]
    if topo == "ring": g += [("CRY", "mu", i, i, n + (i + 1) % n) for i in range(n)]
    if topo == "all": g += [("CRY", "mu", (i, j), i, n + j) for i in range(n) for j in range(n)]
    return g

def cross_edges(n, topo):
    return {(c, t) for k, _, _, c, t in gate_list(n, topo) if k == "CRY"}

def n_circuit_params(n, topo):
    return int(sum(int(np.prod(s)) for s in param_shapes(n, topo).values()))

def run_circuit(params, ts, tb, topo):
    n = ts.shape[1]; st = L.product_state(jnp.concatenate([ts, tb], 1))
    for kind, pn, idx, c, t in gate_list(n, topo):
        if kind == "CNOT": st = L.cnot(st, c, t)
        elif kind == "CRY": st = L.cry(st, c, t, params[pn][idx])
        else: st = jnp.moveaxis(jnp.moveaxis(st, t + 1, -1) @ L.ry_mat(params[pn][idx]).T, -1, t + 1)
    return st.reshape(st.shape[0], -1)

def signs(n):
    N = 2 * n; idx = np.arange(2 ** N); z = 1 - 2 * ((idx[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    return jnp.asarray((z[:, :n] * z[:, n:]).astype(np.float32)), jnp.asarray(z.astype(np.float32))      # (2^N,n) ZZ pairs, (2^N,2n) single-Z

def observables(psi, n, sg):
    """zz_i=<Z_si Z_bi>; QECS=mean zz_i; centred C_i=zz_i-<Z_si><Z_bi>; S_ent = von Neumann entropy (bits) of the behaviour register."""
    zzs, zs = sg; p = psi ** 2; zz = p @ zzs; zm = p @ zs; zsi, zbi = zm[:, :n], zm[:, n:]
    return dict(zz=zz, qecs=zz.mean(1), cqecs=(zz - zsi * zbi).mean(1), sent=L.s_ent(psi, n))
