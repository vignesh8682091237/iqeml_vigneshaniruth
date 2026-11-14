"""Figure 1 (clear left-to-right flow, input -> output). Matches the implemented IQEL-NET (n = 4 per register, ring topology)."""
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arial", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
fig, ax = plt.subplots(figsize=(14.6, 6.0)); ax.set_xlim(0, 14.6); ax.set_ylim(-0.1, 5.9); ax.axis("off")
def card(x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top", fontsize=10, weight="bold")
    ax.text(x + w/2, y + h - 0.62, body, ha="center", va="top", fontsize=8.2, linespacing=1.35)
def arrow(x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333",
                                 linestyle=("--" if dashed else "-")))
def num(x, y, n):
    ax.add_patch(plt.Circle((x, y), 0.17, color="#1565C0")); ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=8.5, weight="bold")
ax.text(7.3, 5.6, "IQEL-NET data flow: from raw OULAD features to a withdrawal / failure risk", ha="center", fontsize=12, weight="bold")
W, H, Y = 1.85, 3.2, 1.3
cols = [
 (0.1, "INPUT", "Behaviour b (9)\nSemantic s (18–19)\nDemographics d (7)\n(checkpoint-limited)", "#E8F5E9", "#2E7D32"),
 (2.2,  "PREPROCESS", "Signed log1p and z-score\n(fitted on training rows)\nTanh projection to\n4 angles per register", "#FFF8E1", "#F9A825"),
 (4.3, "ENCODE", "RY rotations on\n8 qubits:\n4 semantic + 4 behaviour", "#FFF3E0", "#EF6C00"),
 (6.4, "QEMF CIRCUIT", "CNOT rings inside\neach register,\ncross-modal CRY gates,\ntrainable RY layer", "#F3E5F5", "#6A1B9A"),
 (8.5, "READOUT", "ZZ_i = <Z_si Z_bi>\nQECS = mean of ZZ_i\n(A5 also S_ent)", "#FFEBEE", "#C62828"),
 (10.6, "CLASSIFIER", "MLP head\n64 - 32 - 2\n(demographics\nconcatenated here)", "#E3F2FD", "#1565C0"),
 (12.7, "OUTPUT", "Risk of\nwithdrawal / failure\n(target T2)", "#FFFDE7", "#F9A825"),
]
for i, (x, t, b, fc, ec) in enumerate(cols):
    w = W if i < len(cols)-1 else 1.7
    card(x, Y, w, H, t, b, fc, ec)
    num(x + 0.22, Y + H + 0.22, i + 1)
for i in range(len(cols)-1):
    x1 = cols[i][0] + W; x2 = cols[i+1][0]
    arrow(x1 + 0.03, Y + H/2, x2 - 0.03, Y + H/2)
# demographics bypass the quantum block (dashed)
ax.add_patch(FancyArrowPatch((0.1 + W/2, Y - 0.12), (10.6 + W/2, Y - 0.12), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#546E7A", linestyle="--"))
ax.text(5.6, Y - 0.42, "dashed: demographics bypass the quantum block and enter at the classifier", fontsize=8, ha="center", color="#546E7A", style="italic")
ax.text(0.2, 0.42, "Solid arrows: data flow.  Numbered blue circles: processing order.  Inputs are classical; the quantum step is the QEMF circuit only.", fontsize=8, color="#333")
ax.text(0.2, 0.08, "Simulated on a classical computer (exact statevector); no quantum hardware used.", fontsize=8, color="#333")
out = Path(__file__).resolve().parents[1] / "manuscript" / "figure1_qemf_architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight"); print(out)
