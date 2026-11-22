"""Análisis y selección de variables.

Filtra covariables en tres etapas y entrega el set final para modelar. Se ejecuta
sobre el conjunto de entrenamiento.

1. Univariado: porcentaje de nulos y outliers (IQR); descarta variables con más de
   `max_null_pct` de nulos o sin variación.
2. Bivariado: ratio de correlación (η²) contra el objetivo; descarta variables con
   η² menor a `min_power`.
3. Multivariado: correlación entre variables; ante pares con |corr| mayor a
   `max_corr`, conserva la de mayor η².
4. Entrega el set de variables finalistas.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_cleaning import IQR_K, iqr_bounds

CATEGORICAL_ENC_COLS = ["merchant_category_enc", "region_enc"]


@dataclass(frozen=True)
class AnalysisConfig:
    """Umbrales de la selección de variables."""

    max_null_pct: float = 90.0
    min_power: float = 0.01
    max_corr: float = 0.6
    corr_method: str = "spearman"
    n_bins: int = 10
    rank_target: bool = True
    iqr_k: float = IQR_K


def univariate_analysis(df: pd.DataFrame, features: list[str], k: float = IQR_K) -> pd.DataFrame:
    """Calcula porcentaje de nulos, porcentaje de outliers IQR y cantidad de valores únicos."""
    rows = []
    for col in features:
        series = df[col]
        valid = series.dropna()
        if len(valid) > 0:
            lower, upper = iqr_bounds(valid, k)
            pct_outliers = ((valid < lower) | (valid > upper)).mean() * 100
        else:
            pct_outliers = np.nan
        rows.append(
            {
                "feature": col,
                "pct_null": series.isna().mean() * 100,
                "pct_outliers": pct_outliers,
                "n_unique": valid.nunique(),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def _bin_feature(series: pd.Series, n_bins: int, categorical: bool) -> pd.Series:
    """Asigna cada valor a un tramo; los nulos forman un tramo propio."""
    if categorical or series.nunique() <= n_bins:
        bins = series.astype("string")
    else:
        bins = pd.qcut(series, q=n_bins, duplicates="drop").astype("string")
    return bins.fillna("NULL")


def predictive_power(
    feature: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    categorical: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Calcula η² y la tabla de media del objetivo por tramo de una variable.

    η² = varianza entre tramos / varianza total del objetivo.
    """
    bins = _bin_feature(feature, n_bins, categorical)
    table = y.groupby(bins).agg(n="count", y_mean="mean")

    overall_mean = y.mean()
    ss_between = (table["n"] * (table["y_mean"] - overall_mean) ** 2).sum()
    ss_total = ((y - overall_mean) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    return eta2, table


def bivariate_analysis(
    df: pd.DataFrame,
    features: list[str],
    y: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calcula el η² de cada variable contra el objetivo."""
    rows = []
    for col in features:
        categorical = col in CATEGORICAL_ENC_COLS
        eta2, _ = predictive_power(df[col], y, n_bins, categorical)
        rows.append({"feature": col, "eta2": eta2})
    return pd.DataFrame(rows).set_index("feature").sort_values("eta2", ascending=False)


def multivariate_analysis(
    df: pd.DataFrame,
    power: pd.Series,
    max_corr: float = 0.6,
    method: str = "spearman",
) -> tuple[list[str], pd.DataFrame]:
    """Selecciona variables por poder predictivo descendente, descartando las correlacionadas con una ya elegida."""
    ordered = power.sort_values(ascending=False).index.tolist()
    corr = df[ordered].corr(method=method).abs()

    selected = []
    dropped = []
    for col in ordered:
        if selected:
            corr_with_selected = corr.loc[col, selected]
            max_value = corr_with_selected.max()
            if max_value > max_corr:
                dropped.append(
                    {
                        "feature": col,
                        "correlated_with": corr_with_selected.idxmax(),
                        "abs_corr": max_value,
                    }
                )
                continue
        selected.append(col)

    dropped_report = pd.DataFrame(dropped, columns=["feature", "correlated_with", "abs_corr"])
    return selected, dropped_report.set_index("feature")


def select_features(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    config: AnalysisConfig = AnalysisConfig(),
) -> dict:
    """Ejecuta las tres etapas de selección y entrega reportes y variables finalistas."""
    univariate = univariate_analysis(df, features, config.iqr_k)
    keep_uni = univariate[
        (univariate["pct_null"] <= config.max_null_pct) & (univariate["n_unique"] > 1)
    ].index.tolist()

    y = df[target].rank(pct=True) if config.rank_target else df[target]
    bivariate = bivariate_analysis(df, keep_uni, y, config.n_bins)
    keep_bi = bivariate[bivariate["eta2"] >= config.min_power].index.tolist()

    final, dropped_corr = multivariate_analysis(
        df, bivariate.loc[keep_bi, "eta2"], config.max_corr, config.corr_method
    )

    return {
        "univariate": univariate,
        "bivariate": bivariate,
        "dropped_corr": dropped_corr,
        "n_initial": len(features),
        "n_after_univariate": len(keep_uni),
        "n_after_bivariate": len(keep_bi),
        "final_features": final,
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_eng import (
        TARGET,
        build_features,
        encode_categoricals,
        get_feature_columns,
    )

    pd.set_option("display.width", 160)

    data = load_data()
    feats = build_features(data)
    cutoff = feats["date"].max() - pd.DateOffset(months=3)
    train = feats[feats["date"] <= cutoff].copy()
    train, _ = encode_categoricals(train)
    feature_cols = get_feature_columns(train)

    result = select_features(train, feature_cols, TARGET)

    print(f"train hasta {cutoff.date()}: {len(train)} filas")
    print("variables iniciales:", result["n_initial"])
    print("tras univariado:", result["n_after_univariate"])
    print("tras bivariado:", result["n_after_bivariate"])
    print("finalistas:", len(result["final_features"]))
    print()
    print("univariado (mayor % nulos):")
    print(result["univariate"].sort_values("pct_null", ascending=False).head(5).round(2))
    print()
    print("variables finalistas:")
    print(result["bivariate"].loc[result["final_features"]].round(4))
