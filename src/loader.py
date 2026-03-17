from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Nigeria-2025-full-data.dta"

# Continuous numeric features kept from the notebook.
CONTINUOUS_COLS = ["l1", "d2", "n3", "k3a", "k3bc", "k3f", "d3c", "d12b", "b5", "b7"]

# Binary yes/no categorical features.
BINARY_COLS = ["k6", "k82a", "b8", "h1", "h8", "c22b", "e6", "e11", "b4"]

# Ordinal business obstacle ratings.
ORDINAL_COLS = ["k30", "c30a", "j30a", "e30"]

SELECTED_COLUMNS = CONTINUOUS_COLS + BINARY_COLS + ORDINAL_COLS

BINARY_MAP = {"Yes": 1, "No": 0}
OBSTACLE_MAP = {
    "No obstacle": 0,
    "Minor obstacle": 1,
    "Moderate obstacle": 2,
    "Major obstacle": 3,
    "Very severe obstacle": 4,
}


def _resolve_path(data_path: str | Path | None) -> Path:
    return DEFAULT_RAW_DATA_PATH if data_path is None else Path(data_path)


def load_raw_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Stata survey file."""
    return pd.read_stata(_resolve_path(data_path))


def select_notebook_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Select the exact feature set used in the notebooks."""
    missing_cols = [col for col in SELECTED_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in raw data: {missing_cols}")
    return df_raw[SELECTED_COLUMNS].copy()


def encode_notebook_features(df_selected: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook encoding rules for binary, ordinal, and numeric columns."""
    df_encoded = df_selected.copy()

    for col in BINARY_COLS:
        df_encoded[col] = df_encoded[col].map(BINARY_MAP)

    for col in ORDINAL_COLS:
        df_encoded[col] = df_encoded[col].map(OBSTACLE_MAP)

    for col in CONTINUOUS_COLS:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce")

    return df_encoded


def load_and_encode_features(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load raw data and return the encoded feature frame."""
    raw = load_raw_data(data_path=data_path)
    selected = select_notebook_columns(raw)
    return encode_notebook_features(selected)
