# V13 experiment report

## Completed (outputs exist and were inspected)
- **E7 leave-one-module-out**: 231 folds (A1 ×21, A2 ×105, A4-ring ×105), 7 modules, 3 checkpoints. Ring below A2 in 7/7 modules at Weeks 4 and 12, 6/7 at Week 8. Ring below A1 by 0.027–0.040. All six ring comparisons Holm-significant.
- **E8 adapted HQIAT-ML (B4)**: 15 valid runs. A first run with the published robust scaling failed at chance level (test AUC 0.540); cause diagnosed as inputs of order 10^6; the failed run is preserved in `results/b4/b4_failed_robust_scaling.csv`. The valid runs use the stable transform (adaptations A12, A13). B4 ≈ A1 (not significant at Weeks 4 and 8; A1 ahead at Week 12, Holm p = 0.024 student, 0.180 module); B4 > A2 and B4 > ring at every checkpoint, Holm p = 0.009.
- **E11 wide-grid controls (Week 12, 5 seeds)**: A2-w 0.784, A2s-w 0.752, A4-ring-w 0.772. The wider grid improves the ring (0.758 → 0.772) and the parameter-matched control (0.730 → 0.752). The earlier "unstable control" conclusion was partly an under-tuning artefact.
- **E10 noise and shots (simulator only)**: ideal 0.758; readout flip up to 10% lowers AUC by 0.014; depolarising up to 20% lowers it by 0.005; shots 10 → AUC 0.699, 100 → 0.749, 1000 → 0.757, 10000 → 0.758. Ideal values reproduce the stored results in 5/5 seeds.
- **E9 qubit scaling (partial)**: A2 at n = 2, 4, 6 (flat, 0.778–0.779); ring at n = 2 (5 seeds, 0.751) and n = 4 (3 seeds, 0.755). Ring at n = 6 and n = 8 not measured.

## Failed or blocked
- Exact HQIAT-ML reproduction: blocked (see V13_PENDING_WORK.md).
- Ring at n = 6: not completed (time budget).

## Did any result contradict earlier findings?
- The wide-grid controls change the parameter-matched comparison: the earlier narrow-grid A2s was under-tuned, so the earlier "unresolved" verdict is superseded by E11. The main conclusion that quantum circuits do not beat classical fusion is unchanged.
- B4 reaches about the same AUC as the random forest, which is above the classical fusion. This does not contradict the earlier results but adds a strong classical comparator.

## Limits
Simulator only. No hardware. No per-gate noise. Single test presentation. Not a claim of quantum advantage.
