"""
Data preprocessing utilities for Solar PV fault diagnosis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file and perform basic validation."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"], infer_datetime_format=True)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list,
                        scaler: StandardScaler | None = None):
    """Standardise feature columns.

    Returns
    -------
    normalized_df : pd.DataFrame  (copy of df with scaled feature columns)
    scaler        : fitted StandardScaler
    """
    df_out = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_out[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    else:
        df_out[feature_cols] = scaler.transform(df[feature_cols].values)
    return df_out, scaler


def create_sequences(X: np.ndarray, y: np.ndarray,
                     seq_len: int):
    """Slide a window of length seq_len over X and y.

    Returns
    -------
    X_seq : (n_samples, seq_len, n_features)
    y_seq : (n_samples,)  — label at last time-step of each window
    """
    n = len(X)
    if n <= seq_len:
        raise ValueError(f"Dataset too short ({n}) for seq_len={seq_len}.")
    X_seq, y_seq = [], []
    for i in range(n - seq_len):
        X_seq.append(X[i: i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split preserving time order (no shuffling)."""
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def validate_data(df: pd.DataFrame, feature_cols: list):
    """Check that required columns exist and contain finite values.

    Returns
    -------
    is_valid : bool
    issues   : list[str]
    """
    issues = []
    for col in feature_cols:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif df[col].isnull().all():
            issues.append(f"Column '{col}' is entirely NaN.")
        elif not np.isfinite(df[col].dropna().values).all():
            issues.append(f"Column '{col}' contains Inf values.")
    if "fault_label" not in df.columns:
        issues.append("Missing 'fault_label' column.")
    return len(issues) == 0, issues


def fill_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Forward-fill then backward-fill numeric feature columns."""
    df_out = df.copy()
    df_out[feature_cols] = (df_out[feature_cols]
                             .ffill()
                             .bfill()
                             .fillna(0.0))
    return df_out
