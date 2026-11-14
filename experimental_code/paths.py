"""Central path configuration. Works on Windows, Linux and macOS (pathlib only)."""
import os
from pathlib import Path
ROOT = Path(os.environ.get("IQEL_ROOT", Path(__file__).resolve().parents[1]))
RAW = Path(os.environ.get("IQEL_RAW", ROOT / "data" / "raw"))          # folder with the 7 OULAD csv files
COMPACT = Path(os.environ.get("IQEL_COMPACT", ROOT / "data" / "compact"))
RESULTS = ROOT / "results"; LOGS = ROOT / "logs"; PREDS = RESULTS / "preds"
for p in (COMPACT, RESULTS, LOGS, PREDS): p.mkdir(parents=True, exist_ok=True)
