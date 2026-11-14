import numpy as np
from paths import COMPACT
import data_lib
for wk in (4, 8, 12):
    a = data_lib.week_arrays(wk); np.savez_compressed(COMPACT / f"week{wk}.npz", **a)
    print(f"week {wk}: n={len(a['y'])} beh={a['beh'].shape[1]} sem={a['sem'].shape[1]} demo={a['demo'].shape[1]} at-risk={a['y'].mean():.3f} presentations={sorted(set(a['pres']))}")
