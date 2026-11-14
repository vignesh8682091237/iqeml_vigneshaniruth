import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pytest, jax.numpy as jnp, pennylane as qml
import qsim_topo as T, qsim_legacy as L

def rand_params(n, topo, rng):
    s = T.param_shapes(n, topo); return {k: jnp.asarray(rng.uniform(-1, 1, v).astype("float32")) for k, v in s.items()}

def pl_reference(topo, n, a, b, P):
    """Independent PennyLane definition written directly from the topology specification (does NOT use gate_list)."""
    N = 2 * n; dev = qml.device("default.qubit", wires=N)
    def gates():
        for i in range(n): qml.RY(a[i], wires=i)
        for i in range(n): qml.RY(b[i], wires=n + i)
        for i in range(n): qml.CNOT(wires=[i, (i + 1) % n])
        for i in range(n): qml.CNOT(wires=[n + i, n + (i + 1) % n])
        if topo in ("pair", "ring"):
            for i in range(n): qml.CRY(float(P["lam"][i]), wires=[i, n + i])
        if topo == "all":
            for i in range(n):
                for j in range(n): qml.CRY(float(P["lam"][i, j]), wires=[i, n + j])
        for q in range(N): qml.RY(float(P["phi"][q]), wires=q)
        if topo == "ring":
            for i in range(n): qml.CRY(float(P["mu"][i]), wires=[i, n + (i + 1) % n])
        if topo == "all":
            for i in range(n):
                for j in range(n): qml.CRY(float(P["mu"][i, j]), wires=[i, n + j])
    @qml.qnode(dev)
    def state(): gates(); return qml.state()
    @qml.qnode(dev)
    def obs():
        gates(); return [qml.expval(qml.PauliZ(i) @ qml.PauliZ(n + i)) for i in range(n)] + [qml.vn_entropy(wires=list(range(n, N)))]
    return np.array(state()), np.array([float(v) for v in obs()])

@pytest.mark.parametrize("topo", T.TOPOS)
@pytest.mark.parametrize("n", [3, 4])
def test_matches_pennylane(topo, n):
    rng = np.random.RandomState(7); B = 5; P = rand_params(n, topo, rng)
    ts = rng.uniform(-3, 3, (B, n)).astype("float32"); tb = rng.uniform(-3, 3, (B, n)).astype("float32")
    psi = T.run_circuit(P, jnp.asarray(ts), jnp.asarray(tb), topo); sg = T.signs(n); ob = T.observables(psi, n, sg)
    for k in range(B):
        ref_state, ref_obs = pl_reference(topo, n, ts[k], tb[k], P)
        assert np.abs(ref_state.imag).max() < 1e-6                                     # all gates are real
        assert np.abs(ref_state.real - np.array(psi[k])).max() < 2e-6
        assert np.abs(ref_obs[:n] - np.array(ob["zz"][k])).max() < 2e-6
        assert abs(ref_obs[n] / np.log(2) - float(ob["sent"][k])) < 5e-5                  # PennyLane returns nats, ours bits

@pytest.mark.parametrize("n", [3, 4])
def test_cross_modal_edges_are_exactly_the_intended_topology(n):
    assert T.cross_edges(n, "intra") == set()
    assert T.cross_edges(n, "pair") == {(i, n + i) for i in range(n)}
    assert T.cross_edges(n, "ring") == {(i, n + i) for i in range(n)} | {(i, n + (i + 1) % n) for i in range(n)}
    assert T.cross_edges(n, "all") == {(i, n + j) for i in range(n) for j in range(n)}
    for topo in T.TOPOS:                                                               # CNOTs never cross registers; same intra-register wiring in every topology
        cn = [(c, t) for k, _, _, c, t in T.gate_list(n, topo) if k == "CNOT"]
        assert all((c < n) == (t < n) for c, t in cn) and len(cn) == 2 * n

def test_intra_is_a_product_state_across_registers():
    rng = np.random.RandomState(1); n = 4; B = 64; P = rand_params(n, "intra", rng); sg = T.signs(n)
    ts = jnp.asarray(rng.uniform(-3, 3, (B, n)).astype("float32")); tb = jnp.asarray(rng.uniform(-3, 3, (B, n)).astype("float32"))
    ob = T.observables(T.run_circuit(P, ts, tb, "intra"), n, sg)
    assert float(jnp.abs(ob["sent"]).max()) < 1e-4            # zero cross-modal entanglement
    assert float(jnp.abs(ob["cqecs"]).max()) < 1e-5           # separable state => centred correlation is exactly zero

@pytest.mark.parametrize("topo", ["pair", "ring", "all"])
def test_cross_modal_topologies_entangle(topo):
    rng = np.random.RandomState(2); n = 4; B = 64; P = rand_params(n, topo, rng); sg = T.signs(n)
    ts = jnp.asarray(rng.uniform(-3, 3, (B, n)).astype("float32")); tb = jnp.asarray(rng.uniform(-3, 3, (B, n)).astype("float32"))
    ob = T.observables(T.run_circuit(P, ts, tb, topo), n, sg)
    assert float(ob["sent"].mean()) > 0.05 and float(jnp.abs(ob["cqecs"]).max()) > 1e-3

def test_ring_equals_original_simulator():
    rng = np.random.RandomState(3); n = 4; B = 16; P = rand_params(n, "ring", rng)
    ts = jnp.asarray(rng.uniform(-3, 3, (B, n)).astype("float32")); tb = jnp.asarray(rng.uniform(-3, 3, (B, n)).astype("float32"))
    new = T.run_circuit(P, ts, tb, "ring"); old = L.run_circuit(P["lam"], P["phi"], P["mu"], ts, tb)
    assert float(jnp.abs(new - old).max()) < 1e-6

def test_parameter_counts_n4():
    assert [T.n_circuit_params(4, t) for t in T.TOPOS] == [8, 12, 16, 40]
