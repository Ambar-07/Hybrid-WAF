"""
Preprocessing utilities for tabular traffic data.

Features:
- Missing value handling
- Safe numeric conversion
- Optional feature normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


EXCLUDED_COLUMNS = {"label", "timestamp", "src_ip", "dst_ip", "protocol"}


@dataclass
class PreprocessConfig:
    normalize: bool = False
    fill_value: float = 0.0


class Preprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._numeric_columns: List[str] = []
        self._mins = {}
        self._maxs = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        work_df = self._prepare_numeric_df(df.copy())
        self._numeric_columns = [
            c for c in work_df.columns if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(work_df[c])
        ]

        for col in self._numeric_columns:
            self._mins[col] = float(work_df[col].min())
            self._maxs[col] = float(work_df[col].max())

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")

        out = self._prepare_numeric_df(df.copy())

        if self.config.normalize:
            for col in self._numeric_columns:
                if col not in out.columns:
                    continue
                mn = self._mins.get(col, 0.0)
                mx = self._maxs.get(col, 1.0)
                if mx == mn:
                    out[col] = 0.0
                else:
                    out[col] = (out[col] - mn) / (mx - mn)
                    out[col] = out[col].clip(0.0, 1.0)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _prepare_numeric_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip spaces from column names for consistent matching.
        df.columns = [str(c).strip() for c in df.columns]

        # Derive simple time fields if a timestamp column exists.
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            if ts.notna().sum() > 0:
                df["epoch_seconds"] = (ts.astype("int64") // 10**9).astype("float64")
                df["hour"] = ts.dt.hour.astype("float64")
                df["minute"] = ts.dt.minute.astype("float64")

                # seconds since the first timestamp in the batch.
                first_valid = ts.dropna().iloc[0]
                df["time_since_start"] = (ts - first_valid).dt.total_seconds().fillna(0.0)

        for col in df.columns:
            if col in EXCLUDED_COLUMNS:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(self.config.fill_value)
        return df
