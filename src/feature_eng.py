"""Creación de covariables.

Construye variables de calendario, rezagos, estadísticos móviles, interacciones y
codificación de categóricas. Toda variable derivada de flujos usa información hasta t-1.

1. Valida que cada comercio tenga una fila por día, sin huecos.
2. Crea variables de calendario.
3. Crea rezagos por comercio de las series base.
4. Crea estadísticos móviles por comercio (media, desviación, mínimo, máximo,
   mediana, coeficiente de variación) e interacciones entre ellos.
5. Crea razones entre covariables acumuladas por ventana.
6. Codifica variables categóricas con `LabelEncoder`.
7. Entrega la lista de columnas aptas como features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

GROUP_COL = "merchant_id"
DATE_COL = "date"
TARGET = "net_cashflow"
BASE_SERIES = ["net_cashflow", "inflows", "outflows", "txn_count"]
LAGS = [1, 7, 14, 30]
WINDOWS = [1, 7, 14, 30, 60, 90]
CATEGORICAL_COLS = ["merchant_category", "region"]
KNOWN_AT_T_COLS = ["is_holiday", "opening_balance"]
SAME_DAY_COLS = ["inflows", "outflows", "txn_count", "closing_balance", "net_cashflow"]

# Configuración de servicio: solo existen net_cashflow y txn_count de los últimos 14 días.
HISTORY_DAYS = 14
SERVING_SERIES = ["net_cashflow", "txn_count"]
SERVING_LAGS = [1, 7, 14]
SERVING_WINDOWS = [7, 14]
SERVING_CATEGORICALS = ["merchant_category", "region"]


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide dos series y entrega NaN cuando el denominador es 0."""
    result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def check_no_gaps(df: pd.DataFrame) -> None:
    """Lanza error si algún comercio tiene días faltantes o fechas desordenadas."""
    diffs = df.groupby(GROUP_COL)[DATE_COL].diff().dropna()
    if not (diffs == pd.Timedelta(days=1)).all():
        raise ValueError("Existen huecos o desorden en las fechas de algún comercio.")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario y de días de pago."""
    df = df.copy()
    date = df[DATE_COL]
    df["day_of_week"] = date.dt.day_of_week
    df["day_of_month"] = date.dt.day
    df["month"] = date.dt.month
    df["is_weekend"] = (date.dt.day_of_week >= 5).astype(int)
    df["days_to_month_end"] = date.dt.days_in_month - date.dt.day
    df["is_month_end"] = (df["days_to_month_end"] == 0).astype(int)
    df["is_day_5"] = (date.dt.day == 5).astype(int)
    df["is_day_15"] = (date.dt.day == 15).astype(int)
    df["is_day_30"] = (date.dt.day == 30).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str] = BASE_SERIES,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    """Agrega rezagos por comercio."""
    df = df.copy()
    grouped = df.groupby(GROUP_COL)
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
    return df


def _shifted(df: pd.DataFrame, col: str) -> pd.Series:
    """Serie desplazada un día por comercio."""
    return df.groupby(GROUP_COL)[col].shift(1)


def _rolling(df: pd.DataFrame, series: pd.Series, window: int):
    """Ventana móvil por comercio de largo fijo sobre una serie."""
    return series.groupby(df[GROUP_COL]).rolling(window, min_periods=window)


def _to_frame_index(result: pd.Series) -> pd.Series:
    """Quita el nivel de grupo del índice de un resultado móvil."""
    return result.reset_index(level=0, drop=True)


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str] = BASE_SERIES,
    windows: list[int] = WINDOWS,
) -> pd.DataFrame:
    """Agrega estadísticos móviles e interacciones por comercio con datos hasta t-1."""
    df = df.copy()
    new_cols = {}
    for col in columns:
        shifted = _shifted(df, col)
        for w in windows:
            if w < 2:
                continue
            roll = _rolling(df, shifted, w)
            mean = _to_frame_index(roll.mean())
            std = _to_frame_index(roll.std())
            min_ = _to_frame_index(roll.min())
            max_ = _to_frame_index(roll.max())
            median = _to_frame_index(roll.median())

            prefix = f"{col}_roll_{w}"
            new_cols[f"{prefix}_mean"] = mean
            new_cols[f"{prefix}_std"] = std
            new_cols[f"{prefix}_min"] = min_
            new_cols[f"{prefix}_max"] = max_
            new_cols[f"{prefix}_median"] = median
            new_cols[f"{prefix}_cv"] = safe_divide(std, mean.abs())
            new_cols[f"{prefix}_min_max"] = safe_divide(min_, max_)
            new_cols[f"{prefix}_max_mean"] = safe_divide(max_, mean)
            new_cols[f"{prefix}_last_mean"] = safe_divide(shifted, mean)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_ratio_features(df: pd.DataFrame, windows: list[int] = WINDOWS) -> pd.DataFrame:
    """Agrega razones entre covariables acumuladas por ventana con datos hasta t-1."""
    df = df.copy()
    sums = {}
    for col in ["net_cashflow", "inflows", "outflows", "txn_count"]:
        shifted = _shifted(df, col)
        for w in windows:
            sums[(col, w)] = _to_frame_index(_rolling(df, shifted, w).sum())

    new_cols = {}
    for w in windows:
        new_cols[f"inflows_outflows_ratio_{w}"] = safe_divide(sums[("inflows", w)], sums[("outflows", w)])
        new_cols[f"avg_ticket_{w}"] = safe_divide(sums[("inflows", w)], sums[("txn_count", w)])
        new_cols[f"outflow_per_txn_{w}"] = safe_divide(sums[("outflows", w)], sums[("txn_count", w)])
        new_cols[f"net_inflows_ratio_{w}"] = safe_divide(sums[("net_cashflow", w)], sums[("inflows", w)])
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    columns: list[str] = CATEGORICAL_COLS,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Codifica categóricas con `LabelEncoder`; categorías no vistas quedan en -1."""
    df = df.copy()
    if encoders is None:
        encoders = {col: LabelEncoder().fit(df[col]) for col in columns}

    for col in columns:
        mapping = {cls: idx for idx, cls in enumerate(encoders[col].classes_)}
        df[col + "_enc"] = df[col].map(mapping).fillna(-1).astype(int)
    return df, encoders


def build_features(
    df: pd.DataFrame,
    series: list[str] = BASE_SERIES,
    lags: list[int] = LAGS,
    windows: list[int] = WINDOWS,
    ratios: bool = True,
) -> pd.DataFrame:
    """Construye las covariables numéricas sobre el DataFrame completo."""
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    check_no_gaps(df)
    df = add_calendar_features(df)
    df = add_lag_features(df, series, lags)
    df = add_rolling_features(df, series, windows)
    if ratios:
        df = add_ratio_features(df, windows)
    return df


def build_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las covariables disponibles al momento de proyectar (historial de 14 días)."""
    return build_features(df, SERVING_SERIES, SERVING_LAGS, SERVING_WINDOWS, ratios=False)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Entrega las columnas aptas como features, excluyendo datos del mismo día."""
    excluded = set(SAME_DAY_COLS) | set(CATEGORICAL_COLS) | {DATE_COL}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


if __name__ == "__main__":
    from src.data_loader import load_data

    data = load_data()
    feats = build_features(data)
    feats, _ = encode_categoricals(feats)
    feature_cols = get_feature_columns(feats)

    print("shape:", feats.shape)
    print("n features:", len(feature_cols))
    print("incluye balance_delta:", "balance_delta" in feature_cols)
    print("incluye columnas del mismo día:", sorted(set(SAME_DAY_COLS) & set(feature_cols)))

    corr = feats[feature_cols].corrwith(feats[TARGET]).abs().sort_values(ascending=False)
    print("top 10 |corr| con el objetivo:")
    print(corr.head(10).round(3))
