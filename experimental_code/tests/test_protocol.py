import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
import data_lib, train_e6 as E
from paths import COMPACT, RAW
_HAVE_DATA = all((COMPACT / f"week{w}.npz").exists() for w in (4, 8, 12)) and (RAW / "studentInfo.csv").exists()
needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="needs locally rebuilt OULAD features (see REPRODUCTION_GUIDE.md section 4); not shipped in the package")

@needs_data
def test_split_masks_are_disjoint_and_complete():
    for wk in (4, 8, 12):
        d = data_lib.load_week(wk); tune, val, full, test = E.split_masks(d)
        assert set(d["pres"][tune]) == {"2013B", "2013J"} and set(d["pres"][val]) == {"2014B"} and set(d["pres"][test]) == {"2014J"}
        assert not (tune & val).any() and not (full & test).any() and (tune | val | test).all()

@needs_data
def test_preprocessing_never_uses_test_rows():
    d = data_lib.load_week(12); tune, val, full, test = E.split_masks(d); a = E.preprocess(d, full)
    d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    for k in ("beh", "sem", "demo"): d2[k][test] = d2[k][test] * 1000 + 12345          # corrupt every test row
    b = E.preprocess(d2, full)
    for k in ("beh", "sem", "demo"): assert np.array_equal(a[k][~test], b[k][~test])      # statistics came from the training rows only

@needs_data
def test_features_use_no_information_after_the_checkpoint(monkeypatch):
    """Delete every event after the checkpoint from the compact tables; the checkpoint features must not change."""
    for wk, t in ((4, 28), (8, 56)):
        full, beh, sem, _ = data_lib.build(t); orig = pd.read_parquet
        def trunc(path, *a, **k):
            df = orig(path, *a, **k); name = pathlib.Path(path).name
            if name in ("c_daily.parquet", "c_act.parquet"): return df[df.date <= t].copy()
            if name == "c_assess.parquet": return df[df.date_submitted <= t].copy()
            if name == "c_reg.parquet": df = df.copy(); df.loc[df.date_unregistration > t, "date_unregistration"] = np.nan; return df
            return df
        monkeypatch.setattr(data_lib.pd, "read_parquet", trunc); cut, _, _, _ = data_lib.build(t); monkeypatch.setattr(data_lib.pd, "read_parquet", orig)
        cols = beh + sem + data_lib.DEMO
        assert full[["code_module", "code_presentation", "id_student"]].equals(cut[["code_module", "code_presentation", "id_student"]])
        assert np.allclose(full[cols].values.astype(float), cut[cols].values.astype(float), equal_nan=True)

def test_a2s_is_parameter_matched_to_the_ring_circuit():
    import jax
    p2s = E.param_report("A2s", E.init("A2s", jax.random.PRNGKey(0), 19, 9, 7)); p4 = E.param_report("A4-ring", E.init("A4-ring", jax.random.PRNGKey(0), 19, 9, 7))
    assert p2s["params_fusion"] == p4["params_fusion"] == 16 and p2s["params_total"] == p4["params_total"]
