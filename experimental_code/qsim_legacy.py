"""Batched real-amplitude statevector simulator for the IQEL-NET QEMF-X circuit (JAX, CPU).
Wires 0..n-1 = semantic register s, wires n..2n-1 = behaviour register b. All gates are real (RY, CNOT, CRY)."""
import numpy as np, jax, jax.numpy as jnp

def ry_mat(a):
    c, s = jnp.cos(a / 2), jnp.sin(a / 2)
    return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)       # (...,2,2)

def product_state(angles):                       # angles (B,N) -> (B,2,...,2)
    st = jnp.stack([jnp.cos(angles[:, 0] / 2), jnp.sin(angles[:, 0] / 2)], -1)
    for k in range(1, angles.shape[1]):
        v = jnp.stack([jnp.cos(angles[:, k] / 2), jnp.sin(angles[:, k] / 2)], -1)
        st = jnp.einsum('b...,bj->b...j', st, v)
    return st

def _apply(st, U, ax):                           # U (2,2) shared over batch, acts on axis ax
    return jnp.moveaxis(jnp.moveaxis(st, ax, -1) @ U.T, -1, ax)

def cry(st, c, t, ang):
    s0, s1 = jnp.take(st, 0, axis=c + 1), jnp.take(st, 1, axis=c + 1)
    t_ax = t + 1 if t < c else t
    return jnp.stack([s0, _apply(s1, ry_mat(ang), t_ax)], axis=c + 1)

def cnot(st, c, t):
    s0, s1 = jnp.take(st, 0, axis=c + 1), jnp.take(st, 1, axis=c + 1)
    t_ax = t + 1 if t < c else t
    return jnp.stack([s0, jnp.flip(s1, axis=t_ax)], axis=c + 1)

def run_circuit(lam, phi, mu, theta_s, theta_b):
    """QEMF-X: RY encoding -> CNOT rings inside each register -> CRY(lam) s_i->b_i -> trainable RY(phi) on all
    qubits -> CRY(mu) s_i->b_{i+1}.  Returns flat real statevector (B, 2^(2n))."""
    n = theta_s.shape[1]
    st = product_state(jnp.concatenate([theta_s, theta_b], 1))
    for i in range(n): st = cnot(st, i, (i + 1) % n)
    for i in range(n): st = cnot(st, n + i, n + (i + 1) % n)
    for i in range(n): st = cry(st, i, n + i, lam[i])
    for q in range(2 * n): st = jnp.moveaxis(jnp.moveaxis(st, q + 1, -1) @ ry_mat(phi[q]).T, -1, q + 1)
    for i in range(n): st = cry(st, i, n + (i + 1) % n, mu[i])
    return st.reshape(st.shape[0], -1)

def zz_signs(n):
    N = 2 * n; idx = np.arange(2 ** N)
    z = 1 - 2 * ((idx[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)
    return jnp.asarray((z[:, :n] * z[:, n:]).astype(np.float32))              # (2^N, n)

def zz_expect(psi, signs):  return (psi ** 2) @ signs                          # (B, n)

def s_ent(psi, n):
    """von Neumann entropy (bits) of the behaviour register after tracing out the semantic register."""
    m = psi.reshape(psi.shape[0], 2 ** n, 2 ** n)
    rho = jnp.einsum('bsi,bsj->bij', m, m)
    ev = jnp.linalg.eigvalsh(rho)
    return -jnp.sum(jnp.where(ev > 1e-9, ev * jnp.log2(jnp.clip(ev, 1e-9, None)), 0.0), -1)
