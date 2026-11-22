"""Pruebas de construcción de covariables."""

import pandas as pd

from src.feature_eng import HISTORY_DAYS, build_serving_features, get_feature_columns
from tests.conftest import make_series


def test_serving_features_match_full_history():
    """Las covariables con 14 días de historial igualan a las calculadas sobre toda la serie."""
    full = make_series()
    target_idx = 60
    expected = build_serving_features(full).iloc[target_idx]

    window = full.iloc[target_idx - HISTORY_DAYS : target_idx + 1].copy()
    window.loc[window.index[-1], ["net_cashflow", "txn_count"]] = None
    actual = build_serving_features(window).iloc[-1]

    cols = get_feature_columns(build_serving_features(full))
    pd.testing.assert_series_equal(actual[cols].astype(float), expected[cols].astype(float), check_names=False)


def test_features_do_not_use_same_day_value():
    """Cambiar el flujo del día proyectado no altera sus covariables."""
    full = make_series()
    changed = full.copy()
    changed.loc[60, "net_cashflow"] = 10**9
    a = build_serving_features(full).iloc[60].drop(["net_cashflow"])
    b = build_serving_features(changed).iloc[60].drop(["net_cashflow"])
    pd.testing.assert_series_equal(a.astype(str), b.astype(str))
