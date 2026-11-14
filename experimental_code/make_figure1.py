"""Figure 1 for manuscript v11: architecture as actually implemented (n = 4 qubits per register, ring topology)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
fig, ax = plt.subplots(figsize=(12.5, 6.6)); ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.6); ax.axis("off")
def box(x, y, w, h, t, fc, ec, fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=1.3)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333"))
ax.text(6.25, 6.35, "IQEL-NET as implemented: QEMF circuit with QECS and entanglement-entropy readouts (n = 4 qubits per register)", ha="center", fontsize=11.5, weight="bold")
box(0.15, 4.7, 2.5, 1.0, "Behaviour features b (9)\nclicks, active days, interactions,\nclick SD, span, assessments due,\nsubmitted, submission rate, lateness", "#E8F5E9", "#2E7D32", 6.9)
box(0.15, 3.2, 2.5, 1.0, "Semantic/metadata features s (18-19)\ndistinct days per VLE activity type\n(categorical; no BERT in these runs)", "#FFF3E0", "#C77700", 6.9)
box(0.15, 1.7, 2.5, 0.9, "Demographics d (7)\nage, education, IMD, sex,\ndisability, attempts, credits", "#ECEFF1", "#546E7A", 7.8)
for y in (5.2, 3.7): arr(2.65, y, 3.15, y)
box(3.15, 4.7, 1.9, 1.0, "signed log1p + z-score\n(fit on training rows)\nW_b: theta_b = pi tanh(W_b b)", "#E8F5E9", "#2E7D32", 7.8)
box(3.15, 3.2, 1.9, 1.0, "signed log1p + z-score\n(fit on training rows)\nW_s: theta_s = pi tanh(W_s s)", "#FFF3E0", "#C77700", 7.8)
for y in (5.2, 3.7): arr(5.05, y, 5.55, y)
ax.add_patch(FancyBboxPatch((5.55, 2.85), 3.55, 3.0, boxstyle="round,pad=0.04,rounding_size=0.1", fc="#F3E5F5", ec="#6A1B9A", lw=1.6))
ax.text(7.32, 5.55, "QEMF quantum circuit (8 qubits, exact simulation)", ha="center", fontsize=8.2, weight="bold")
steps = ["1  RY(theta) encoding: s on 4 qubits, b on 4 qubits", "2  fixed CNOT ring inside each register", "3  CRY(lambda_i)  s_i -> b_i  (trainable)", "4  RY(phi_q) on all 8 qubits (trainable)", "5  CRY(mu_i)  s_i -> b_(i+1)  (trainable)"]
for k, s in enumerate(steps): ax.text(5.7, 5.15 - 0.38 * k, s, fontsize=7.5, va="center")
ax.text(7.32, 3.12, "Topology ablation: intra (no cross-modal gates) | pair (step 3 only) |\nring (steps 3+5, shown) | all (all s_i -> b_j in steps 3 and 5)", ha="center", fontsize=7.1, style="italic")
arr(9.1, 4.4, 9.6, 4.4)
box(9.6, 4.25, 2.7, 1.35, "Readouts\nZZ_i = <Z_si Z_bi>, i = 1..4\nQECS = mean_i ZZ_i (correlation)\nS_ent = von Neumann entropy of\nbehaviour register (entanglement)", "#FFEBEE", "#C62828", 7.0)
arr(2.65, 2.15, 9.6, 2.15); ax.text(6.0, 2.25, "demographics bypass the quantum block", fontsize=7.4, ha="center", style="italic")
arr(10.95, 4.25, 10.95, 3.15)
box(9.6, 1.55, 2.7, 1.55, "Classifier head (one MLP)\ninput: [QECS, ZZ_1..4] + d\n(A5 also S_ent, without gradient)\n64 - 32 - 2, focal + smoothing\n(+ contrastive in A5)", "#E3F2FD", "#1565C0", 7.8)
arr(10.95, 1.55, 10.95, 0.95); box(9.6, 0.35, 2.7, 0.6, "Risk of withdrawal / failure (T2)", "#FFFDE7", "#F9A825", 8.2, True)
box(0.15, 0.2, 8.9, 1.1, "Control arms with the same projections and head: A2 = classical bilinear fusion (198 fusion parameters);\nA2s = parameter-matched classical fusion (16 parameters); A3 = circuit with a learned linear readout of the 256 amplitudes;\nA1n = MLP on all features without bottleneck.", "#FAFAFA", "#9E9E9E", 7.2)
out = Path(__file__).resolve().parents[1] / "manuscript" / "figure1_qemf_architecture.png"; fig.savefig(out, dpi=200, bbox_inches="tight"); print(out)
