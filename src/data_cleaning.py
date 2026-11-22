"""Limpieza de datos.

Elimina duplicados, excluye columnas, reporta outliers y acota valores extremos con la regla IQR.

0. Excluye columnas configuradas (`exclude_columns`), por ejemplo `merchant_id`.
1. Elimina filas duplicadas exactas y duplicados de la llave (`merchant_id`, `date`).
2. Reporta outliers por columna con la regla IQR.
3. Calcula límites IQR por comercio sobre un conjunto de referencia.
4. Acota (clip) las columnas a esos límites.
5. Expone el acotamiento como opción configurable con `OutlierConfig`.
"""

from dataclasses import dataclass, field

import pandas as pd

KEY_COLUMNS = ["merchant_id", "date"]
OUTLIER_COLUMNS = ["inflows", "outflows", "txn_count", "net_cashflow"]
CAP_COLUMNS = ["inflows", "outflows", "txn_count"]
IQR_K = 1.5


@dataclass(frozen=True)
class OutlierConfig:
    """Parámetros del acotamiento de outliers."""

    enabled: bool = False
    columns: list[str] = field(default_factory=lambda: list(CAP_COLUMNS))
    k: float = IQR_K
    group_col: str | None = "merchant_id"


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Elimina duplicados exactos y duplicados de llave, conservando la primera fila."""
    n_initial = len(df)
    df = df.drop_duplicates()
    n_exact = n_initial - len(df)

    n_before_key = len(df)
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    n_key = n_before_key - len(df)

    report = {
        "n_initial": n_initial,
        "n_exact_duplicates": n_exact,
        "n_key_duplicates": n_key,
        "n_final": len(df),
    }
    return df, report


def exclude_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Elimina las columnas indicadas que existan en el DataFrame."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def iqr_bounds(series: pd.Series, k: float = IQR_K) -> tuple[float, float]:
    """Calcula los límites inferior y superior IQR de una serie."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_report(
    df: pd.DataFrame,
    columns: list[str] = OUTLIER_COLUMNS,
    k: float = IQR_K,
    group_col: str | None = "merchant_id",
) -> pd.DataFrame:
    """Cuenta outliers IQR por columna, globales y por grupo."""
    rows = []
    for col in columns:
        lower, upper = iqr_bounds(df[col], k)
        n_global = ((df[col] < lower) | (df[col] > upper)).sum()

        row = {
            "column": col,
            "q1": df[col].quantile(0.25),
            "q3": df[col].quantile(0.75),
            "lower": lower,
            "upper": upper,
            "n_outliers_global": n_global,
            "pct_outliers_global": n_global / len(df) * 100,
        }

        if group_col is not None:
            bounds = df.groupby(group_col)[col].transform(lambda s: iqr_bounds(s, k)[0])
            upper_g = df.groupby(group_col)[col].transform(lambda s: iqr_bounds(s, k)[1])
            n_group = ((df[col] < bounds) | (df[col] > upper_g)).sum()
            row["n_outliers_by_group"] = n_group
            row["pct_outliers_by_group"] = n_group / len(df) * 100

        rows.append(row)

    return pd.DataFrame(rows).set_index("column")


def fit_iqr_bounds(
    df: pd.DataFrame,
    columns: list[str] = CAP_COLUMNS,
    k: float = IQR_K,
    group_col: str | None = "merchant_id",
) -> pd.DataFrame:
    """Calcula límites IQR por grupo (o globales) para cada columna."""
    records = []
    if group_col is None:
        groups = [(None, df)]
    else:
        groups = df.groupby(group_col)

    for key, group in groups:
        for col in columns:
            lower, upper = iqr_bounds(group[col], k)
            records.append({"group": key, "column": col, "lower": lower, "upper": upper})

    return pd.DataFrame(records)


def clip_iqr(
    df: pd.DataFrame,
    bounds: pd.DataFrame,
    group_col: str | None = "merchant_id",
) -> pd.DataFrame:
    """Acota las columnas a los límites IQR calculados con `fit_iqr_bounds`."""
    df = df.copy()
    for col, col_bounds in bounds.groupby("column"):
        if group_col is None:
            lower = col_bounds["lower"].iloc[0]
            upper = col_bounds["upper"].iloc[0]
        else:
            lower = df[group_col].map(col_bounds.set_index("group")["lower"])
            upper = df[group_col].map(col_bounds.set_index("group")["upper"])
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def cap_outliers(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: OutlierConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Acota train y test con límites IQR ajustados en train, si `config.enabled`."""
    if not config.enabled:
        return train, test, None

    bounds = fit_iqr_bounds(train, config.columns, config.k, config.group_col)
    train_capped = clip_iqr(train, bounds, config.group_col)
    test_capped = clip_iqr(test, bounds, config.group_col)
    return train_capped, test_capped, bounds


if __name__ == "__main__":
    from src.data_loader import load_data

    data = load_data()
    data, dup_report = remove_duplicates(data)
    print("duplicados:", dup_report)
    print(outlier_report(data))

    cutoff = data["date"].max() - pd.DateOffset(months=3)
    train_df = data[data["date"] <= cutoff]
    test_df = data[data["date"] > cutoff]

    for enabled in (False, True):
        cfg = OutlierConfig(enabled=enabled)
        train_out, test_out, _ = cap_outliers(train_df, test_df, cfg)
        n_train = (train_out[cfg.columns] != train_df[cfg.columns]).sum().sum()
        n_test = (test_out[cfg.columns] != test_df[cfg.columns]).sum().sum()
        print(f"enabled={enabled}: acotados train={n_train}, test={n_test}")
