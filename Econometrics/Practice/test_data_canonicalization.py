"""
Unit tests for Enum-based data_canonicalization.py
"""
import pytest
import pandas as pd
from pathlib import Path
from utils import data_canonicalization as dc
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

def test_canonicalize_rows_basic():
    schema = {
        "color": {"type": "categorical", "enum": Color},
        "flag": {"type": "boolean"},
        "count": {"type": "int"},
        "score": {"type": "float"},
        "name": {"type": "string"}
    }
    raw = [
        {"color": "red", "flag": 1, "count": 1, "score": 1.1, "name": "a"},
        {"color": "green", "flag": 0, "count": 2, "score": 2.2, "name": "b"},
        {"color": "blue", "flag": None, "count": None, "score": None, "name": "c"},
        {"color": "red", "flag": True, "count": 4, "score": "3.3", "name": None},
        {"color": None, "flag": False, "count": "5", "score": "bad", "name": "e"}
    ]
    rows = dc.canonicalize_rows(raw, schema)
    print("test_canonicalize_rows_basic results:")
    for i, r in enumerate(rows):
        print(f"Row {i}: color={r.color}, flag={r.flag}, count={r.count}, score={r.score}, name={r.name}")
    assert rows[0].color == Color.RED
    assert rows[1].color == Color.GREEN
    assert rows[2].color == Color.BLUE
    assert rows[4].color is None
    assert rows[0].flag is True
    assert rows[1].flag is False
    assert rows[2].flag is None
    assert rows[4].score is None

def test_canonical_to_pandas_and_parquet(tmp_path):
    class Cat(Enum):
        A = "a"
        B = "b"
        C = "c"
    schema = {
        "cat": {"type": "categorical", "enum": Cat},
        "flag": {"type": "boolean"},
        "val": {"type": "float"}
    }
    raw = [
        {"cat": "a", "flag": True, "val": 1.0},
        {"cat": "b", "flag": False, "val": 2.0},
        {"cat": "c", "flag": None, "val": 3.0},
        {"cat": "a", "flag": True, "val": 4.0}
    ]
    rows = dc.canonicalize_rows(raw, schema)
    df = dc.canonical_to_pandas(rows, schema)
    print("test_canonical_to_pandas_and_parquet DataFrame:")
    print(df)
    assert set(df["cat"].cat.categories) == {"a", "b", "c"}
    out_path = tmp_path / "test_enum.parquet"
    dc.save_canonical_parquet(rows, schema, out_path)
    loaded_rows = dc.load_canonical_parquet(out_path, schema)
    print("Loaded rows from parquet:")
    for i, r in enumerate(loaded_rows):
        print(f"Row {i}: cat={r.cat}, flag={r.flag}, val={r.val}")
    assert [r.cat for r in loaded_rows] == [Cat.A, Cat.B, Cat.C, Cat.A]

def test_edge_cases_enum():
    class Fruit(Enum):
        APPLE = "apple"
        BANANA = "banana"
    schema = {
        "fruit": {"type": "categorical", "enum": Fruit},
        "flag": {"type": "boolean"}
    }
    raw = [
        {"fruit": "apple", "flag": 1},
        {"fruit": "banana", "flag": 0},
        {"fruit": "pear", "flag": "bad"},
        {"fruit": None, "flag": None}
    ]
    rows = dc.canonicalize_rows(raw, schema)
    print("test_edge_cases_enum results:")
    for i, r in enumerate(rows):
        print(f"Row {i}: fruit={r.fruit}, flag={r.flag}")
    assert rows[2].fruit is None  # Unknown category
    assert rows[2].flag is None  # Non-boolean value

def run_all_tests():
    print("Running test_canonicalize_rows_basic...")
    test_canonicalize_rows_basic()
    print("Passed.")
    print("Running test_canonical_to_pandas_and_parquet...")
    test_canonical_to_pandas_and_parquet(Path("./"))
    print("Passed.")
    print("Running test_edge_cases_enum...")
    test_edge_cases_enum()
    print("Passed.")
    print("All tests passed.")

if __name__ == "__main__":
    run_all_tests()
