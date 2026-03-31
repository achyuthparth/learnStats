
"""
Data Canonicalization Utilities (Python/Enum-based)

Defines strict canonical form using Python types and enums, not pandas dtypes.
Provides conversion to/from pandas DataFrame for compute.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

# --- Example: Enum for a categorical variable ---
# Users should define their own enums for each categorical variable
# Example:
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

# --- Canonical Data Classes ---
class CanonicalRow:
    def __init__(self, data: Dict[str, Any], schema: Dict[str, Any]):
        self._schema = schema
        for k, v in data.items():
            setattr(self, k, self._coerce(k, v))
    def _coerce(self, key, value):
        meta = self._schema[key]
        t = meta["type"]
        if t == "categorical":
            enum_cls = meta["enum"]
            if value is None or value == "":
                return None
            try:
                return enum_cls(value)
            except ValueError:
                return None
        elif t == "boolean":
            if value in (True, 1, "1", "True", "true"): return True
            if value in (False, 0, "0", "False", "false"): return False
            return None
        elif t == "int":
            try: return int(value)
            except: return None
        elif t == "float":
            try: return float(value)
            except: return None
        elif t == "string":
            return str(value) if value is not None else None
        else:
            return value
    def as_dict(self):
        return {k: getattr(self, k, None) for k in self._schema}

# --- Canonicalization Functions ---
def canonicalize_rows(raw_rows: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[CanonicalRow]:
    return [CanonicalRow(row, schema) for row in raw_rows]

def canonical_to_pandas(rows: List[CanonicalRow], schema: Dict[str, Any]) -> pd.DataFrame:
    # For compute: categorical columns as pandas.Categorical, etc.
    data = []
    for row in rows:
        d = row.as_dict()
        for k, meta in schema.items():
            if meta["type"] == "categorical":
                d[k] = d[k].value if d[k] is not None else None
        data.append(d)
    df = pd.DataFrame(data)
    for k, meta in schema.items():
        if meta["type"] == "categorical":
            df[k] = pd.Categorical(df[k], categories=[e.value for e in meta["enum"]])
        elif meta["type"] == "boolean":
            df[k] = df[k].astype("boolean")
        elif meta["type"] == "int":
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
        elif meta["type"] == "float":
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Float64")
        elif meta["type"] == "string":
            df[k] = df[k].astype("string")
    return df

def save_canonical_parquet(rows: List[CanonicalRow], schema: Dict[str, Any], path: Path):
    df = canonical_to_pandas(rows, schema)
    df.to_parquet(path, index=False)

def load_canonical_parquet(path: Path, schema: Dict[str, Any]) -> List[CanonicalRow]:
    df = pd.read_parquet(path)
    raw_rows = df.to_dict(orient="records")
    return canonicalize_rows(raw_rows, schema)
