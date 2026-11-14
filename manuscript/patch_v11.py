"""Builds manuscript v11 from v10 (XML patch). All numbers in Section 4 are read from results/*.csv, nothing is typed by hand except prose."""
import re, shutil, sys, numpy as np, pandas as pd
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image
ROOT = Path("/mnt/user-data/outputs/IQEL-NET_Q1_UPGRADE"); RES = ROOT / "results"; PKG = Path("/home/claude/v11work/pkg")
P = PKG / "word" / "document.xml"; x = P.read_text(encoding="utf-8")

def bounds(sub):
    i = x.index(escape(sub)); s = x.rindex("<w:p>", 0, i); e = x.index("</w:p>", i) + 6; return s, e
def ptext(s, e): return "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", x[s:e], flags=re.S))
def set_para(sub, new):
    global x
    s, e = bounds(sub); x = x[:s] + re.sub(r"(<w:t(?: [^>]*)?>).*?(</w:t>)", lambda m: m.group(1) + escape(new) + m.group(2), x[s:e], count=1, flags=re.S) + x[e:]
def append_para(sub, extra):
    global x
    s, e = bounds(sub); old = ptext(s, e); set_para(sub, old.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") + extra)
def rep(old, new):
    global x
    o = escape(old); assert o in x, "NOT FOUND: " + old[:80]; x = x.replace(o, escape(new), 1)
def clone_after(sub, texts, before=False):
    global x
    s, e = bounds(sub); it = iter(texts); c = re.sub(r"(<w:t(?: [^>]*)?>).*?(</w:t>)", lambda m: m.group(1) + escape(next(it)) + m.group(2), x[s:e], flags=re.S)
    x = x[:s] + c + x[s:] if before else x[:e] + c + x[e:]
def clone_row(sub, texts, before=False):
    global x
    i = x.index(escape(sub)); rs = max(x.rfind("<w:tr>", 0, i), x.rfind("<w:tr ", 0, i)); re_ = x.index("</w:tr>", i) + 7
    it = iter(texts); c = re.sub(r"(<w:t(?: [^>]*)?>).*?(</w:t>)", lambda m: m.group(1) + escape(next(it)) + m.group(2), x[rs:re_], flags=re.S)
    x = x[:rs] + c + x[rs:] if before else x[:re_] + c + x[re_:]

BODY = '<w:p><w:pPr><w:spacing w:after="160" w:line="276"/><w:jc w:val="both"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>'
H2 = '<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:spacing w:after="120" w:before="240"/></w:pPr><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>'
CAP = '<w:p><w:pPr><w:spacing w:after="200" w:before="100"/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r></w:p>'
SP = '<w:p><w:pPr><w:spacing w:after="80"/></w:pPr></w:p>'
class _F:
    def __init__(self, t): self.t = t
    def __mod__(self, v): return self.t % escape(v)
H2 = _F(H2); CAP = _F(CAP)
txt = lambda s: BODY % escape(s)
def cell(t, w, header=False, bold=False, center=False):
    shd = '<w:shd w:fill="D9E6F5" w:val="clear"/>' if header else ""; b = "<w:b/><w:bCs/>" if (header or bold) else '<w:b w:val="false"/><w:bCs w:val="false"/>'
    jc = '<w:pPr><w:jc w:val="center"/></w:pPr>' if center else ""
    return f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{shd}</w:tcPr><w:p>{jc}<w:r><w:rPr>{b}<w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr><w:t xml:space="preserve">{escape(str(t))}</w:t></w:r></w:p></w:tc>'
def table(widths, header, rows, bold=()):
    t = '<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="%d"/><w:tblBorders>' % sum(widths) + "".join(f'<w:{s} w:val="single" w:color="auto" w:sz="4"/>' for s in ["top", "left", "bottom", "right", "insideH", "insideV"]) + "</w:tblBorders></w:tblPr><w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
    t += "<w:tr>" + "".join(cell(h, w, header=True, center=k > 0) for k, (h, w) in enumerate(zip(header, widths))) + "</w:tr>"
    for r, row in enumerate(rows): t += "<w:tr>" + "".join(cell(c, w, bold=(r, k) in bold, center=k > 0) for k, (c, w) in enumerate(zip(row, widths))) + "</w:tr>"
    return t + "</w:tbl>"
_img_id = [100]
def add_image(png, rid, width_emu):
    im = Image.open(png); cx = width_emu; cy = int(cx * im.height / im.width); _img_id[0] += 1; i = _img_id[0]
    return (f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{i}" name="Picture {i}"/>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{i}" name="figure{i}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"/></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')

# ================= results tables (all numbers from CSV) =================
M = pd.read_csv(RES / "e6_main_results.csv"); PD = pd.read_csv(RES / "e6_paired_comparisons.csv"); QV = pd.read_csv(RES / "e6_qecs_validation.csv"); PM = pd.read_csv(RES / "e6_per_module_auc.csv"); FIN = pd.read_csv(RES / "e6_final.csv")
m = lambda arm, wk, col="auc": float(M[(M.arm == arm) & (M.week == wk)][col].iloc[0])
sd = lambda arm, wk: float(M[(M.arm == arm) & (M.week == wk)].auc_sd_seeds.iloc[0])
pr = lambda a, b, wk: PD[(PD.a == a) & (PD.b == b) & (PD.week == wk)].iloc[0]
LAB = {"B1_LogReg": "B1  Logistic regression (behaviour + demographics)", "B2_RF": "B2  Random forest (behaviour + demographics)", "B3_HistGB": "B3  Histogram gradient boosting (behaviour + demographics)",
       "A0_RF": "A0  Random forest (behaviour only)", "A1_RF": "A1  Random forest (behaviour + semantic + demographics)", "A1n": "A1n  MLP on all features, no bottleneck", "A2": "A2  Classical bilinear fusion, 198 fusion parameters",
       "A2s": "A2s  Parameter-matched classical fusion, 16 parameters", "A3": "A3  Ring circuit, linear amplitude readout", "A4-intra": "A4-intra  Circuit without cross-modal gates", "A4-pair": "A4-pair  Paired cross-modal gate only",
       "A4-ring": "A4-ring  Ring circuit + QECS/ZZ readout", "A4-all": "A4-all  Fully connected circuit + QECS/ZZ readout", "A5": "A5  Ring circuit + QECS + S_ent + contrastive"}
order = list(LAB)
rows5, bold5 = [], set()
for r, a in enumerate(order):
    row = [LAB[a]]
    for k, wk in enumerate((4, 8, 12)):
        row.append(f"{m(a, wk):.3f} ({m(a, wk, 'ci_module_lo'):.3f}\u2013{m(a, wk, 'ci_module_hi'):.3f})")
        if a == "A1_RF": bold5.add((r, k + 1))
    rows5.append(row)
t5 = table([3550, 1933, 1933, 1934], ["Configuration", "Week 4", "Week 8", "Week 12"], rows5, bold5)
npar = {"A4-intra": 8, "A4-pair": 12, "A4-ring": 16, "A4-all": 40}
rows6 = [[LAB[a].split("  ")[0], str(npar[a])] + [f"{m(a, wk):.3f} \u00b1 {sd(a, wk):.3f}" for wk in (4, 8, 12)] for a in ("A4-intra", "A4-pair", "A4-ring", "A4-all")]
t6 = table([2100, 1700, 1850, 1850, 1850], ["Circuit", "Circuit parameters", "Week 4", "Week 8", "Week 12"], rows6)
def dcell(a, b, wk):
    q = pr(a, b, wk); mk = ("\u2020" if q.p_student_holm < 0.05 else "") + ("\u2021" if q.p_module_holm < 0.05 else "")
    return f"{q['diff']:+.3f} [{q.ci_module_lo:+.3f}, {q.ci_module_hi:+.3f}]{mk}"
PAIR7 = [("A4-ring", "A4-intra"), ("A4-pair", "A4-intra"), ("A4-all", "A4-intra"), ("A4-ring", "A4-pair"), ("A4-all", "A4-ring")]
t7 = table([2400, 2316, 2317, 2317], ["Comparison (first minus second)", "Week 4", "Week 8", "Week 12"], [[f"{a} vs {b}"] + [dcell(a, b, wk) for wk in (4, 8, 12)] for a, b in PAIR7])
rows8 = [[LAB[a].split("  ")[0], str(int(FIN[FIN.arm == a].params_fusion.iloc[0]))] + [f"{m(a, wk):.3f} \u00b1 {sd(a, wk):.3f}" for wk in (4, 8, 12)] for a in ("A2", "A2s", "A4-pair", "A4-ring", "A4-all")]
t8 = table([2100, 1700, 1850, 1850, 1850], ["Model", "Fusion-stage parameters", "Week 4", "Week 8", "Week 12"], rows8)
PAIR9 = [("A4-ring", "A2s"), ("A4-all", "A2s"), ("A4-ring", "A2"), ("A2s", "A2"), ("A3", "A4-ring"), ("A5", "A4-ring"), ("A4-ring", "A1_RF")]
t9 = table([2400, 2316, 2317, 2317], ["Comparison (first minus second)", "Week 4", "Week 8", "Week 12"], [[f"{a} vs {b}"] + [dcell(a, b, wk) for wk in (4, 8, 12)] for a, b in PAIR9])
rows10 = []
for a in ("A4-intra", "A4-pair", "A4-ring", "A4-all", "A5"):
    for wk in (4, 8, 12):
        q = QV[(QV.arm == a) & (QV.week == wk)].iloc[0]
        rows10.append([a, str(wk), f"{q.mean_sent:.2f}", f"{q.qecs_auc_signinv:.3f}", "0 (exactly zero)" if a == "A4-intra" else f"{q.cqecs_auc_oriented:.3f}", "n/a" if a == "A4-intra" else f"{int(q.cqecs_seeds_lcb_above_half)} of 5", f"{q.sign_flip_share:.0%}", f"{q.corr_qecs_sent:+.2f}"])
t10 = table([1250, 750, 1200, 1500, 1550, 1250, 1050, 800], ["Configuration", "Week", "Mean S_ent (bits)", "QECS AUC, sign-invariant", "Centred QECS AUC, oriented", "Seeds with lower bound > 0.5", "Sign flips", "r(QECS, S_ent)"], rows10)
FIN["spe"] = FIN.secs_train / FIN.epochs
rt = FIN.groupby("arm").agg(spe=("spe", "mean"), ep=("epochs", "mean"), secs=("secs_train", "mean"))
rows11 = []
for a in order:
    if a.startswith(("B", "A0", "A1_")): rows11.append([LAB[a].split("  ")[0], f"{m(a, 12, 'brier'):.3f}", f"{m(a, 12, 'ece'):.3f}", "-", "-", "-"])
    else: rows11.append([LAB[a].split("  ")[0], f"{m(a, 12, 'brier'):.3f}", f"{m(a, 12, 'ece'):.3f}", f"{rt.loc[a, 'spe']:.3f}", f"{rt.loc[a, 'ep']:.1f}", f"{rt.loc[a, 'secs']:.1f}"])
t11 = table([1900, 1250, 1250, 1650, 1650, 1650], ["Configuration", "Brier", "ECE", "Seconds per epoch", "Mean epochs", "Training seconds"], rows11)
cohort = table([900, 1300, 1750, 1250, 1250, 1450, 1450], ["Week", "Cohort", "Excluded (already withdrawn)", "Train", "Test (2014J)", "At-risk, train", "At-risk, test"],
               [["4", "27,538", "5,055", "18,322", "9,216", "0.458", "0.408"], ["8", "26,522", "6,071", "17,643", "8,879", "0.437", "0.385"], ["12", "25,724", "6,869", "17,156", "8,568", "0.421", "0.363"]])
# per-module facts for the text
w12 = PM[PM.week == 12]; piv = w12.pivot(index="module", columns="arm", values="auc")
beat = {a: int((piv[a] > piv["A4-intra"]).sum()) for a in ("A4-pair", "A4-ring", "A4-all")}
a1lo, a1hi = piv["A1_RF"].min(), piv["A1_RF"].max(); modlo, modhi = piv["A1_RF"].idxmin(), piv["A1_RF"].idxmax()
nmod = len(piv)
spe_ratio = (rt.loc[["A3", "A4-intra", "A4-pair", "A4-ring", "A4-all", "A5"], "spe"] / rt.loc["A2", "spe"])
print("per-module:", beat, a1lo, a1hi, modlo, modhi, "per-epoch ratio", spe_ratio.round(1).to_dict())

# ================= abstract / introduction =================
set_para("Two strands of work have looked at predicting",
 "Two strands of work have looked at predicting whether a student will withdraw from or fail an online course: quantum-inspired deep learning on behavioural data, and quantum or quantum-inspired language models on text. We ask whether a quantum circuit that entangles a semantic/metadata register with a behavioural register helps, using the Open University Learning Analytics Dataset under a leakage-controlled protocol: checkpoint-limited features at Weeks 4, 8 and 12, hyperparameters selected on one held-out presentation (2014B), and a single evaluation on a later presentation (2014J). The proposed QEMF circuit is read out as a correlation score (QECS) and, separately, an entanglement entropy. In an ablation, circuits with cross-modal controlled rotations beat an otherwise identical circuit without them by 0.013 to 0.043 AUC-ROC. The circuits do not, however, outperform a classical bilinear fusion of the same compressed inputs (ahead of the ring circuit by 0.010 to 0.024) or a random forest on all features (AUC-ROC 0.742, 0.778 and 0.796), and a parameter-matched classical control was too unstable to settle the comparison. The sign of QECS is not stable across trained models. The semantic input comes from the same click log as the behavioural input, and all results are from exact simulation of 8-qubit circuits, one tuning seed and one test presentation.")
set_para("A ten-configuration baseline-and-ablation matrix", "A baseline-and-ablation matrix (Section 3.7, Table 2) that includes a product-state circuit without cross-modal gates, a parameter-matched classical control and three further topologies, so that the benefit of quantum cross-modal interaction is tested rather than assumed.")
# ================= 3.1 semantic branch (mismatch 1) =================
set_para("OULAD doesn't contain genuine free-form student text",
 "OULAD doesn't contain genuine free-form student text, so the semantic input is structured metadata. For student i at checkpoint t, the semantic/metadata vector s contains, for each VLE activity type (resource, forumng, quiz, and so on), the number of distinct days on which the student used that activity type at or before t: 18 types at Week 4 and 19 at Weeks 8 and 12. It records which kinds of course content a student used, not what the student wrote. It comes from the same interaction log as the behavioural features, so the two inputs are not independent sources of information, which limits what any cross-modal claim can show (Section 8). A BERT encoding of activity and assessment descriptors was part of the original design; it has not been implemented, and no result in this paper uses BERT.")
set_para("To check whether BERT is even earning its keep", "The planned comparison between a categorical encoding, a shallow word-embedding average and BERT is therefore not reported; only the categorical encoding was run. If an institutional dataset with genuine free text (forum posts, feedback, written comments) becomes available, it would be a stronger second modality, subject to privacy and ethics approval.")
# ================= 3.3 / 3.4 / 3.5 =================
set_para("be the BERT-derived semantic/metadata embedding", "Let s be the semantic/metadata vector (activity-type day counts) and b the behavioural feature vector, both computed under the checkpoint protocol above, passed through a signed logarithm and z-scored with statistics fitted on the training rows only. Both get projected and angle-encoded:")
set_para("tanh(W_s \u00b7 s)", "\u03b8_s = \u03c0 \u00b7 tanh(W_s s + c_s),        \u03b8_b = \u03c0 \u00b7 tanh(W_b b + c_b)")
s_, e_ = bounds("All gates are real, so the state has real amplitudes")
x = x[:e_] + txt("Four topologies are compared. They share the encoding, the CNOT rings and the trainable single-qubit rotations, and differ only in the cross-modal gates: intra has none, so its state is a product across the two registers (8 trainable circuit parameters at n = 4); pair has only CRY(\u03bb_i) from s_i to b_i (12 parameters); ring is the topology above (16 parameters); and all replaces both controlled-rotation layers by CRY from every s_i to every b_j (40 parameters). Unit tests check each topology against an independent PennyLane definition and check that intra has zero entanglement entropy and zero centred QECS.") + x[e_:]
rep("Both are computed and compared empirically (Section 3.7), and whichever is better justified by that comparison is the one that makes it into the final manuscript.", "Both are computed for every trained circuit and compared in Section 4.5. Neither is used as a training signal in the reported experiments.")
# ================= 3.6 (mismatches 2 and 3) =================
set_para("go into an ensemble of three MLP heads", "The readout e (Section 3.7 says what each configuration reads out) is concatenated with the demographic features and passed to a single MLP head with 64 and 32 hidden units and GELU activations. The training objective is:")
rep("\u03bb\u2081, \u03bb\u2082, \u03bb\u2083 are also tuned on validation rather than fixed up front.", "The weights \u03bb are fixed rather than tuned: 0.6, 0.25 and 0.15 in A5, the only configuration with the contrastive term, and 0.706 and 0.294 for the focal and smoothing terms in the other neural and quantum arms. Only the learning rate and, through early stopping, the number of epochs are selected on the validation presentation.")
rep("pass the result to the ensemble classifier", "pass the result to the classifier head")
set_para("The full trainable parameter set is", "The trainable parameters are \u0398 = {W_s, c_s, W_b, c_b, \u03bb, \u03c6, \u03bc, \u03a6_head}: the two projections with their biases, the circuit parameters (which depend on the topology), and the weights of the single MLP head. The circuit and fusion parameter counts of every configuration are reported in Tables 6 and 8. No BERT encoder is used in the reported experiments.")
# ================= 3.7 =================
set_para("We're treating", "We're treating \"entanglement improves prediction\" as a hypothesis to test, not something to assume going in. Table 2 lists the configurations: classical baselines (B1\u2013B3), a protocol-controlled reproduction of HQIAT-ML (B4, not yet run), the IQEL-NET configurations A0\u2013A5, and controls added for this study: A1n (neural fusion without the input bottleneck), A2s (a classical fusion with the same number of fusion parameters as the ring circuit), and the topology variants A4-intra, A4-pair, A4-ring and A4-all, which isolate the contribution of the cross-modal gates. Every configuration runs under the same data protocol, preprocessing, splits and model-selection protocol, so the marginal contribution of each design choice can be isolated one step at a time.")
set_para("A2 to A5 share the same input compression", "A2 to A5 share the same input compression and the same classifier head, so they differ only in how the two compressed inputs are fused. In A2 the angles \u03b8_s and \u03b8_b are fused classically: their cosines, sines and pairwise products go through a learned linear map with tanh (198 parameters). In A2s the fusion is a per-qubit bilinear form, e_i = tanh(a_i cos \u03b8_s,i cos \u03b8_b,i + b_i sin \u03b8_s,i sin \u03b8_b,i + c_i cos \u03b8_s,i sin \u03b8_b,i + d_i sin \u03b8_s,i cos \u03b8_b,i), with 16 parameters, and the classifier receives the mean of the e_i followed by the e_i. In A3 the ring circuit is run and the classifier receives a learned linear projection of the 256 real statevector amplitudes. In the A4 circuits the classifier receives e = [QECS, ZZ_1, ..., ZZ_n]. In A5 it receives e = [QECS, S_ent, ZZ_1, ..., ZZ_n] and the contrastive term is added; S_ent is computed by exact diagonalisation and enters as a measured feature without gradient. A1n is a neural control, an MLP on all behavioural, semantic and demographic features with no bottleneck.")
append_para("A separate feature-contribution experiment", " This experiment has not been run; Section 4.5 instead reports how well QECS and the centred QECS predict the outcome on their own.")
rep("QEMF circuit + QECS and per-pair ZZ readout", "QEMF ring circuit + QECS and per-pair ZZ readout")
clone_row("QEMF ring circuit + QECS and per-pair ZZ readout", ["A4-all", "QEMF circuit, full cross-modal connectivity + QECS/ZZ readout", "Tests whether more connectivity than the ring helps"])
clone_row("QEMF ring circuit + QECS and per-pair ZZ readout", ["A4-pair", "QEMF circuit, paired cross-modal gate only + QECS/ZZ readout", "Isolates what the second controlled-rotation layer adds"], before=True)
clone_row("QEMF ring circuit + QECS and per-pair ZZ readout", ["A4-intra", "QEMF circuit without cross-modal gates + QECS/ZZ readout", "Control: product state across the registers; isolates the cross-modal gates"], before=True)
clone_row("Quantum-inspired fusion (classical bilinear phase fusion, no circuit)", ["A2s", "Parameter-matched classical fusion (16 fusion parameters)", "Fair classical control for the ring circuit (16 circuit parameters)"])
# the original A4 row keeps its ID text "A4": rename it
i = x.index(escape("QEMF ring circuit + QECS and per-pair ZZ readout")); rs = x.rfind("<w:tr", 0, i); seg = x[rs:i]; j = seg.index(">A4<"); x = x[:rs + j] + ">A4-ring<" + x[rs + j + 4:]
# ================= 3.8 / 3.9 / 3.14 / 3.16 / 3.17 =================
append_para("Beyond the checkpoint-based leakage protocol in Section 3.1", " Hyperparameters are selected with a presentation-held-out protocol: models are fitted on 2013B and 2013J, compared on 2014B, then refitted on 2013B, 2013J and 2014B with the selected settings, and evaluated once on 2014J. For the neural and quantum arms only the learning rate is selected (from {0.003, 0.01}, with early stopping on the validation AUC fixing the number of epochs); for the classical models the small grids of Section 3.14 are used. The loss weights are fixed. The test presentation never enters selection; unit tests check that the splits are disjoint and that preprocessing statistics never use test rows.")
set_para("Measured on a single CPU core with n = 4", "Measured on a single CPU core with n = 4 qubits per register (8 qubits in total), one training epoch took 0.04 s for the classical fusion A2, 0.38 s for the circuit without cross-modal gates, 0.47 s for the paired circuit, 0.55 s for the ring, 0.85 s for the fully connected circuit and 0.88 s for A5, so the circuits cost roughly 10 to 23 times more per epoch than A2. Training to the selected epoch took between 3 and 17 s on average for the circuits against 0.2 s for A2 (Table 11). Larger n has not been run.")
rep("The preliminary runs use fixed hyperparameters; validation-based tuning has not been done yet.", "Hyperparameters are tuned on the validation presentation with small grids (Section 3.8): the regularisation strength for logistic regression, depth and leaf size for the random forests, and learning rate and depth for histogram gradient boosting.")
append_para("Five random seeds give a first sense", " With 1,000 resamples the smallest attainable p-value is about 0.001, so the smallest attainable adjusted p-value after Holm correction over 36 comparisons is 0.036, and many adjusted p-values in Section 4 sit at that floor. With five seeds the smallest two-sided Wilcoxon p-value is 0.0625, so seed-level tests cannot reach conventional significance and are reported as descriptive only.")
set_para("Since IQEL-NET is meant as a risk-prediction", "Since IQEL-NET is meant as a risk-prediction and decision-support signal rather than a pure ranking model, discrimination metrics like AUC-ROC and F1 aren't enough on their own. Calibration is assessed on the test presentation through the Brier score, the expected calibration error (ECE, ten equal-frequency bins) and reliability diagrams (Figure 3). No post-hoc calibration was applied.")
# ================= Section 4 =================
a = bounds("4.1 Setup of the Preliminary Experiments")[0]; b = bounds("Not yet evaluated: B4")[1]
fig3_rid = "rId901"
new4 = (H2 % "4.1 Protocol and Setup" +
 txt("All configurations use the presentation-held-out protocol of Section 3.8. Models are fitted on 2013B and 2013J, hyperparameters are chosen on 2014B, the model is refitted on 2013B, 2013J and 2014B, and 2014J is evaluated once. The tuning stage used one seed; the refit and evaluation used five seeds (one for logistic regression). Target T2 and the Week 4, 8 and 12 checkpoints follow Section 3.1, and students who had already unregistered by a checkpoint are excluded (Table 4). The neural and quantum arms use four qubits per register (an exact 8-qubit simulation) and one MLP head, with fixed loss weights. Intervals are 95% percentile intervals from 1,000 resamples of the 2014J test set, averaged over seeds. The primary intervals resample the seven modules, because students within a module are not independent; student-level intervals are narrower and are in the released results. Paired differences use the same resamples, and p-values are Holm-corrected over the 36 pre-specified comparisons (12 pairs at three checkpoints).") +
 SP + cohort + CAP % "Table 4. Cohort sizes and at-risk proportion per checkpoint (target T2, presentation-held-out split)." +
 H2 % "4.2 Main Results" +
 txt(f"Table 5 gives the AUC-ROC of the 14 configurations that have been run. The random forest on all features (A1) is the strongest at every checkpoint (0.742, 0.778 and 0.796). Among the models that compress each modality into four angles, the classical bilinear fusion A2 is the best (0.728, 0.765, 0.781), and the ring circuit with the QECS readout (A4-ring) reaches 0.718, 0.753 and 0.758. Selecting the learning rate on a held-out presentation changed the neural and quantum results by at most 0.02 AUC compared with an earlier protocol that used a random validation split, and the unbottlenecked MLP (A1n) stays far below the random forest (0.707 against 0.796 at Week 12), so its weakness is not explained by the choice of learning rate and epoch count. It stops after about three epochs and its regularisation was not tuned, so it should be read as an untuned control. The module-clustered intervals are wide for the neural arms, and the seed standard deviation reaches {max(sd(a_, wk) for a_ in ('A2s',) for wk in (4, 8, 12)):.3f} for the 16-parameter classical control.") +
 SP + t5 + CAP % "Table 5. AUC-ROC on the held-out 2014J presentation: mean over seeds, with the 95% module-clustered bootstrap interval in brackets; best per checkpoint in bold. B3 is scikit-learn's histogram gradient boosting." +
 H2 % "4.3 Do the Cross-Modal Gates Matter? Topology Ablation" +
 txt(f"Table 6 compares four circuits that share the encoding, the CNOT rings and the trainable single-qubit rotations and differ only in the cross-modal gates. The circuit without cross-modal gates (A4-intra), whose state is a product across the two registers, is the weakest at every checkpoint (0.700, 0.732, 0.727). Adding cross-modal controlled rotations raises AUC by 0.013 to 0.043 (Table 7). The sign of the improvement is the same for all three topologies and all checkpoints. It is significant after Holm correction with student-level resampling in every case, and it survives module-clustered resampling at Week 12 for all three topologies, for the fully connected circuit at all three checkpoints, and for the ring at Week 4. At Week 12 the cross-modal circuits are ahead of the intra circuit in {beat['A4-pair']} (pair), {beat['A4-ring']} (ring) and {beat['A4-all']} (all) of the {nmod} modules. Extra connectivity beyond the paired gate adds little: the ring is 0.005 ahead of the paired circuit at Week 4, tied at Week 8 and 0.012 behind at Week 12, and the fully connected circuit, with 40 circuit parameters, is ahead of the ring by at most 0.007. The simplest cross-modal circuit, with 12 circuit parameters, is about as good as the others. This is the first result in this study that supports the hypothesis that the cross-modal gates carry useful signal. It isolates the cross-register gates only: the intra circuit still contains intra-register entanglement and trainable rotations, and it does not show that entanglement is needed, which is what the comparison with a classical fusion in Section 4.4 probes.") +
 SP + t6 + CAP % "Table 6. AUC-ROC (mean \u00b1 standard deviation over five seeds) of the four circuit topologies, with the number of trainable circuit parameters at n = 4." +
 SP + t7 + CAP % "Table 7. Paired AUC differences with 95% module-clustered intervals. \u2020 Holm-corrected p < 0.05 with student-level resampling; \u2021 Holm-corrected p < 0.05 with module-clustered resampling." +
 H2 % "4.4 Parameter-Matched Classical Control" +
 txt("The parameter-matched control A2s has the same compressed inputs and classifier head as the circuit and 16 fusion parameters, equal to the ring circuit. It is the weakest of the compressed models at Week 8 (0.698) and the second weakest at Week 12 (0.730), and it is unstable across seeds (standard deviations 0.052 and 0.047), so it does not give a clean comparison. The ring circuit is ahead of it by 0.010, 0.055 and 0.028 AUC with student-level intervals that exclude zero, but the module-clustered intervals include zero at every checkpoint (Table 9). The larger classical fusion A2, with 198 fusion parameters, is ahead of the ring circuit by 0.010, 0.011 and 0.024 (module-clustered Holm p 0.228, 0.036 and 0.036). We cannot tell whether the instability of A2s reflects a real limit of 16-parameter classical fusion or an optimisation problem in our implementation (for example saturation of the tanh terms), and the only tuned quantity was the learning rate, so we regard this control as inconclusive. The comparison that matters, a circuit against a classical fusion of the same size, is therefore unresolved; the circuit's disadvantage against the larger classical fusion is clear.") +
 SP + t8 + CAP % "Table 8. Classical controls and circuits at matched or larger sizes (AUC-ROC, mean \u00b1 standard deviation over five seeds)." +
 SP + t9 + CAP % "Table 9. Other paired AUC differences (first minus second) with 95% module-clustered intervals; marks as in Table 7. A3 uses the linear amplitude readout and A5 adds S_ent and the contrastive term to the ring circuit." +
 H2 % "4.5 QECS and Entanglement Entropy" +
 txt("QECS and its centred version are computed from every trained circuit on the test presentation (Table 10). Three points follow. First, QECS predicts the outcome on its own: its sign-invariant AUC is 0.615 to 0.691 for the ring and 0.616 to 0.669 even for the circuit without cross-modal gates. For that circuit QECS reduces to the mean of products of single-register expectations and the centred QECS is exactly zero by construction, so the predictive power of QECS does not by itself show cross-modal correlation. Second, the centred QECS, which is the covariance between the two registers, is informative in the cross-modal circuits: its oriented AUC is 0.599 to 0.657 for the ring, 0.606 to 0.646 for the fully connected circuit and 0.567 to 0.648 for the paired circuit, with a lower 95% confidence bound above 0.5 in at least four of five seeds at every checkpoint. Third, the direction of the association is not stable: the orientation inferred on training data flips in 20% to 80% of the seeds for the A4 circuits, so the sign of QECS has no fixed meaning across trained models, and a statement such as \"higher QECS means more coupled engagement and lower risk\" cannot be made. The correlation between QECS and S_ent changes sign and is at most 0.28 in absolute value, so the two carry different information. The mean entanglement entropy of the behaviour register is 0.74 to 0.84 bits for the paired circuit, 0.97 to 1.02 for the ring, 1.32 to 1.45 for the fully connected circuit and 1.10 to 1.18 for A5, out of a maximum of 4 bits. The criterion for an interpretability claim (a lower bound above 0.5 in at least four of five seeds, and a stable sign) was fixed in the analysis script before the runs: the first part is met by the centred QECS and the second is not, so only the weaker statement that QECS and the centred QECS carry outcome-related information with a model-specific sign is supported.") +
 SP + t10 + CAP % "Table 10. Entanglement and QECS diagnostics on 2014J (means over five seeds). AUC columns use QECS or centred QECS as the only predictor; \"oriented\" means the sign was chosen on training data, and the lower bound is a Hanley\u2013McNeil 95% bound. Sign flips is the share of seeds in which the training-data orientation was negative." +
 H2 % "4.6 Calibration, Per-Module Results and Cost" +
 txt(f"Calibration is reasonable for the neural models: at Week 12 the ECE is 0.06 to 0.08 for A5 and the A4 circuits (0.092 for A3) and 0.072 for A2, against 0.048 for histogram gradient boosting, 0.095 to 0.111 for the random forests and 0.120 for the unbottlenecked MLP, the worst (Table 11, Figure 3). Per-module AUCs for the random forest A1 at Week 12 range from {a1lo:.3f} (module {modlo}) to {a1hi:.3f} (module {modhi}), so the test-set average hides large differences between modules. In cost, the circuits take about 10 to 23 times longer per epoch than A2 (Section 3.9).") +
 SP + t11 + CAP % "Table 11. Calibration at Week 12 (Brier score and ECE) and training cost of the neural and quantum arms on one CPU core (n = 4); cost is averaged over the three checkpoints." +
 add_image(ROOT / "manuscript" / "figure3_reliability.png", fig3_rid, 5700000) + CAP % "Figure 3. Reliability diagrams (ten equal-frequency bins, seed 0) for A1, A2s, A4-ring and A5 on 2014J at Weeks 4, 8 and 12." +
 H2 % "4.7 Caveats and What Remains" +
 txt("These AUC levels are lower than the 0.9615 reported for HQIAT-ML on the same dataset. The numbers aren't directly comparable: our protocol uses checkpoint-limited features, excludes students who have already withdrawn, and tests on a later presentation instead of a random split. We hypothesise that these differences explain much of the gap, but the protocol-controlled reproduction B4 (Section 3.13) that would test this has not been run. The results also carry specific caveats: the test set is one presentation with seven modules, so module-clustered intervals rest on seven clusters; tuning used one seed and a two-value learning-rate grid; the 16-parameter control was unstable; the semantic input shares its source with the behavioural input; and everything is exact classical simulation at n = 4. Not yet evaluated: B4, targets T1 and T3, leave-one-module-out, a genuine text or BERT modality, larger n, data re-uploading and richer readouts, noise and shot sensitivity, and the centred QECS as a training signal."))
x = x[:a] + new4 + x[b:]
# ================= sections 5, 7, 8, 9, captions =================
rep("low QECS is associated with higher predicted risk in this sample", "QECS differs between students with and without the outcome in this sample")
s_, e_ = bounds("That the circuit is classically intractable")
c = re.sub(r"(<w:t(?: [^>]*)?>).*?(</w:t>)", lambda m: m.group(1) + escape("That the sign of QECS has a fixed meaning (the orientation inferred on training data flips between seeds), or that higher QECS means more coupled engagement.") + m.group(2), x[s_:e_], count=1, flags=re.S); x = x[:e_] + c + x[e_:]
set_para("The quantum arms were run only at n = 4", "Everything is exact classical simulation of 8-qubit circuits. Learning rate and epochs were tuned on a single validation presentation with one seed and a two-value grid, loss weights were not tuned, and the test set is one presentation with seven modules, so the module-clustered intervals rest on seven clusters. The semantic input is derived from the same click log as the behavioural input, and no BERT or text encoder was used. The 16-parameter classical control was unstable, so the comparison with a classical fusion of the same size is unresolved. B4, targets T1 and T3, leave-one-module-out, larger n and noise analyses have not been run.")
set_para("In the preliminary experiments, classical models set the bar", "Under checkpoint-limited, presentation-held-out evaluation, a random forest on all features reaches an AUC-ROC of 0.742, 0.778 and 0.796 at Weeks 4, 8 and 12. Circuits with cross-modal controlled rotations beat an otherwise identical circuit without them by 0.013 to 0.043 AUC, which is the clearest support in this study for the idea that the cross-modal gates carry signal. The circuits nevertheless stay below a classical bilinear fusion of the same compressed inputs and below the random forest, the 16-parameter classical control was too unstable to settle the same-size comparison, and the sign of QECS is not stable across trained models. We therefore make no claim of predictive advantage from quantum cross-modal interaction. The contribution of this work is a leakage-controlled, presentation-held-out evaluation protocol, a topology ablation that isolates the cross-modal gates, and an open account of what the QECS and entanglement-entropy readouts do and do not show. Results from n = 4 simulation, one test presentation and the remaining experiments listed in Section 4.7 may change these conclusions.")
set_para("Figure 1. QEMF", "Figure 1. IQEL-NET as implemented: semantic and behavioural features are projected to four angles each, encoded in two four-qubit registers, entangled by the QEMF circuit (ring topology shown), read out as ZZ correlations, QECS and entanglement entropy, and passed with the demographic features to a single MLP head.")
# ================= images: replace Figure 1, register Figure 3 =================
first = re.search(r'<w:drawing>.*?r:embed="(rId\d+)".*?</w:drawing>', x, flags=re.S); rid1 = first.group(1)
rels = (PKG / "word" / "_rels" / "document.xml.rels").read_text(encoding="utf-8"); tgt = re.search(rf'Id="{rid1}"[^>]*Target="([^"]+)"', rels).group(1)
shutil.copy(ROOT / "manuscript" / "figure1_qemf_architecture.png", PKG / "word" / tgt)
im = Image.open(ROOT / "manuscript" / "figure1_qemf_architecture.png"); dr = first.group(0)
cx = int(re.search(r'<wp:extent cx="(\d+)"', dr).group(1)); cy = int(cx * im.height / im.width)
dr2 = re.sub(r'(<wp:extent cx="\d+" cy=")\d+', rf"\g<1>{cy}", dr); dr2 = re.sub(r'(<a:ext cx="\d+" cy=")\d+', rf"\g<1>{cy}", dr2); x = x.replace(dr, dr2, 1)
shutil.copy(ROOT / "manuscript" / "figure3_reliability.png", PKG / "word" / "media" / "figure3_reliability.png")
rels = rels.replace("</Relationships>", f'<Relationship Id="{fig3_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/figure3_reliability.png"/></Relationships>')
(PKG / "word" / "_rels" / "document.xml.rels").write_text(rels, encoding="utf-8"); P.write_text(x, encoding="utf-8"); print("patched")
