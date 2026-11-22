"""Fixtures compartidas: modelo pequeño entrenado con datos sintéticos."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.export_model import model_columns
from src.feature_eng import (
    SERVING_CATEGORICALS,
    build_serving_features,
    encode_categoricals,
)


def make_series(days: int = 80) -> pd.DataFrame:
    """Serie sintética de un comercio con estacionalidad semanal."""
    rng = np.random.default_rng(67)
    dates = pd.date_range("2026-01-01", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "merchant_id": "M-TEST",
            "merchant_category": "retail",
            "region": "Sur",
            "is_holiday": 0,
            "opening_balance": 1_000_000.0,
            "net_cashflow": 100_000 + 50_000 * (dates.day_of_week.values < 5) + rng.normal(0, 10_000, days),
            "txn_count": rng.integers(50, 150, days),
        }
    )


@pytest.fixture(scope="session")
def model_file(tmp_path_factory) -> str:
    """Entrena un modelo mínimo con el mismo flujo de features y lo guarda en disco."""
    df = build_serving_features(make_series())
    df, encoders = encode_categoricals(df, columns=SERVING_CATEGORICALS)
    columns = model_columns(df)
    df = df.dropna(subset=columns)
    model = RandomForestRegressor(n_estimators=10, random_state=67).fit(df[columns], df["net_cashflow"])
    bundle = {"model": model, "columns": columns, "encoders": encoders, "version": "test", "trained_until": "2026-03-21"}
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(bundle, path)
    return str(path)
