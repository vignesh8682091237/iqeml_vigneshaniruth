"""Second pass over manuscript v11: remove statements that still describe things that were not implemented or not run."""
import re
from pathlib import Path
from xml.sax.saxutils import escape
P = Path("/home/claude/v11work/pkg/word/document.xml"); x = P.read_text(encoding="utf-8")
def rep(old, new):
    global x
    o = escape(old); assert o in x, "NOT FOUND: " + old[:80]; x = x.replace(o, escape(new), 1)
def span_rep(start, end, new):
    global x
    a = x.index(escape(start)); b = x.index(escape(end), a) + len(escape(end)); x = x[:a] + escape(new) + x[b:]
def set_para(sub, new):
    global x
    i = x.index(escape(sub)); s = x.rindex("<w:p>", 0, i); e = x.index("</w:p>", i) + 6
    x = x[:s] + re.sub(r"(<w:t(?: [^>]*)?>).*?(</w:t>)", lambda m: m.group(1) + escape(new) + m.group(2), x[s:e], count=1, flags=re.S) + x[e:]
rep("3.6.2 Trainable Parameters and BERT Configuration", "3.6.2 Trainable Parameters")
rep("Projecting a high-dimensional BERT embedding into a small qubit register can compress away information, so we compare n \u2208 {4, 8, 12} qubits per modality,",
    "Projecting the semantic and behavioural feature vectors (about 19 and 9 dimensions) into four angles each can compress away information, so we plan to compare n \u2208 {4, 8, 12} qubits per register,")
set_para("To make this independently reproducible",
 "To make this independently reproducible, the released repository contains the exact package versions, the configuration file, the logs, every test-set prediction and the code for all steps, from the raw OULAD files to the tables in Section 4. It records the software framework, the simulator backend, the training hardware, all random seeds, the optimiser and early-stopping rule, the hyperparameter grids, the qubit count and circuit depth of each configuration, the gate-parameter initialisation, the projection-matrix dimensions, the normalisation and every preprocessing step. The experiments use a custom batched real-amplitude statevector simulator written in JAX, checked against PennyLane's default.qubit to within 1e-6 on statevectors, ZZ expectations and entanglement entropy, on a single CPU core. The semantic features are fixed by the released feature code (the activity types and the day-count definition); if a BERT or text encoder is added later, its template, model, tokenizer and pooling will be reported.")
rep("Main results use T2; T1 and T3 are robustness checks.", "Main results use T2; T1 and T3 are planned robustness checks that have not been run yet.")
rep("are reported as robustness checks", "are planned as robustness checks that have not been run yet")
rep("We run and report two evaluations separately:", "We plan two evaluations, reported separately, of which only the first has been run:")
rep("4. Experimental Framework and Preliminary Results", "4. Experiments and Results")
span_rep("For every configuration in Table 2 (B1\u2013B4, A0\u2013A5), the full study reports", "leave-one-module-out evaluation within OULAD;",
         "For every configuration that has been run, we report AUC-ROC, PR-AUC, F1, recall, the Brier score and calibration, with bootstrap confidence intervals and paired comparisons. Generalisation is checked through a held-out later presentation (leave-one-module-out has not been run);")
P.write_text(x, encoding="utf-8"); print("patched b")
