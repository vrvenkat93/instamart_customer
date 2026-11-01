import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

NULL_MARKERS = {"", "na", "n/a", "null", "none", "nan", "unknown"}

def norm_header(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

def safe_table_name(name: str) -> str:
    name = norm_header(Path(name).stem)
    return name or "dataset"

def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)
    out = df.copy()
    out.columns = [norm_header(c) for c in out.columns]
    out = out.replace(list(NULL_MARKERS), pd.NA)
    duplicates = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)
    for col in out.columns:
        if any(token in col for token in ("date", "time", "timestamp", "created", "updated")):
            parsed = pd.to_datetime(out[col], errors="coerce", utc=True)
            if parsed.notna().any():
                out[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    quality = {
        "rows_before": before,
        "rows_after": len(out),
        "exact_duplicates_removed": duplicates,
        "missing_cells_after_cleaning": int(out.isna().sum().sum()),
        "columns": list(out.columns),
    }
    return out, quality

def resolve_column(df: pd.DataFrame, canonical: str, aliases: dict) -> str | None:
    candidates = [norm_header(x) for x in aliases.get(canonical, [canonical])]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    # Conservative fuzzy match: exact token set, not arbitrary substring.
    for col in df.columns:
        if col.replace("_", "") in {x.replace("_", "") for x in candidates}:
            return col
    return None

def load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}. Run the supplied generate_data.py first.")
    result = {}
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [norm_header(c) for c in df.columns]
        key = safe_table_name(path.name)
        if key in result:
            key = f"{key}_{len(result)+1}"
        result[key] = df
    return result

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def sqlite_connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)
