"""Checkpoint-limited feature construction (identical logic to the original data_lib/run_classical_matrix build())."""
import numpy as np, pandas as pd
from paths import RAW, COMPACT
key = ["code_module", "code_presentation", "id_student"]
DEMO = ["age", "edu", "imd", "male", "disab", "num_of_prev_attempts", "studied_credits"]
CHECKPOINT_DAY = {4: 28, 8: 56, 12: 84}

def _static():
    info = pd.read_csv(RAW / "studentInfo.csv")
    info = info[info.final_result.isin(["Withdrawn", "Fail", "Pass", "Distinction"])].copy()
    info["y"] = info.final_result.isin(["Withdrawn", "Fail"]).astype(int)                       # target T2
    info["age"] = info.age_band.map({"0-35": 0, "35-55": 1, "55<=": 2})
    info["edu"] = info.highest_education.map({"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2, "HE Qualification": 3, "Post Graduate Qualification": 4})
    info["imd"] = info.imd_band.map({f"{i}-{i+10}%": i // 10 for i in range(0, 100, 10)})
    info["male"] = (info.gender == "M").astype(int); info["disab"] = (info.disability == "Y").astype(int)
    sched = pd.read_csv(RAW / "assessments.csv"); sched["date"] = pd.to_numeric(sched["date"], errors="coerce")
    return info, sched

def build(t):
    info, sched = _static()
    daily, act = pd.read_parquet(COMPACT / "c_daily.parquet"), pd.read_parquet(COMPACT / "c_act.parquet")
    asm, reg = pd.read_parquet(COMPACT / "c_assess.parquet"), pd.read_parquet(COMPACT / "c_reg.parquet")
    df = info.merge(reg[key + ["date_unregistration"]], on=key, how="left")
    n0 = len(df); df = df[~(df.date_unregistration <= t)].copy()             # already withdrawn by t -> not early warning
    d = daily[daily.date <= t].groupby(key).agg(total_clicks=("clicks", "sum"), n_active_days=("date", "nunique"), n_interactions=("rows", "sum"),
                                                click_std=("clicks", "std"), first=("date", "min"), last=("date", "max")).reset_index()
    d["study_span"] = d["last"] - d["first"]; d = d.drop(columns=["first", "last"])
    n_due = sched[sched.date <= t].groupby(["code_module", "code_presentation"]).size().reset_index(name="n_due")
    s = asm[(asm.date <= t) & (asm.date_submitted <= t)].copy(); s["late"] = s.date_submitted - s.date      # submission occurrence only, never the score
    s = s.groupby(key).agg(n_sub=("id_assessment", "count"), mean_late=("late", "mean")).reset_index()
    sem = act[act.date <= t].groupby(key + ["activity_type"]).size().unstack(fill_value=0).add_prefix("sem_").reset_index()
    df = df.merge(d, on=key, how="left").merge(s, on=key, how="left").merge(sem, on=key, how="left").merge(n_due, on=["code_module", "code_presentation"], how="left")
    df["n_due"] = df.n_due.fillna(0)
    df["sub_rate"] = np.where(df.n_due > 0, df.n_sub.fillna(0) / df.n_due.replace(0, np.nan), 0)
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    beh = ["total_clicks", "n_active_days", "n_interactions", "click_std", "study_span", "n_due", "n_sub", "sub_rate", "mean_late"]
    df[beh + sem_cols] = df[beh + sem_cols].fillna(0); df[DEMO] = df[DEMO].fillna(df[DEMO].median())
    return df, beh, sem_cols, n0

def week_arrays(wk):
    df, beh, sem, n0 = build(CHECKPOINT_DAY[wk])
    return dict(beh=df[beh].values.astype("float32"), sem=df[sem].values.astype("float32"), demo=df[DEMO].values.astype("float32"),
                y=df.y.values.astype("int32"), pres=df.code_presentation.values.astype(str), module=df.code_module.values.astype(str),
                beh_cols=np.array(beh), sem_cols=np.array(sem), demo_cols=np.array(DEMO), n_all=n0)

def load_week(wk):
    d = np.load(COMPACT / f"week{wk}.npz", allow_pickle=True); return {k: d[k] for k in d.files}
