"""Step 1 of the pipeline: reduce the 10.6M-row OULAD click log to compact per-student tables (events up to day 84 only)."""
import pandas as pd
from paths import RAW, COMPACT
key = ["code_module", "code_presentation", "id_student"]
vle = pd.read_csv(RAW / "vle.csv", usecols=["id_site", "activity_type"])
sv = pd.read_csv(RAW / "studentVle.csv", usecols=key + ["id_site", "date", "sum_click"])
sv = sv[sv["date"] <= 84].merge(vle, on="id_site", how="left")             # nothing after Week 12 is ever needed
sv.groupby(key + ["date"]).agg(clicks=("sum_click", "sum"), rows=("sum_click", "size")).reset_index().to_parquet(COMPACT / "c_daily.parquet")
sv.groupby(key + ["activity_type", "date"]).size().reset_index(name="n")[key + ["activity_type", "date"]].to_parquet(COMPACT / "c_act.parquet")
sa = pd.read_csv(RAW / "studentAssessment.csv", usecols=["id_assessment", "id_student", "date_submitted"])
a = pd.read_csv(RAW / "assessments.csv"); a["date"] = pd.to_numeric(a["date"], errors="coerce")
sa.merge(a, on="id_assessment", how="left").to_parquet(COMPACT / "c_assess.parquet")
reg = pd.read_csv(RAW / "studentRegistration.csv"); reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
reg.to_parquet(COMPACT / "c_reg.parquet"); print("compact tables written to", COMPACT)
