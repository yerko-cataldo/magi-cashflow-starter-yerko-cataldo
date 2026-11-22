"""Evaluación y selección de modelos.

Calcula métricas de regresión sobre el conjunto de test para cada modelo entrenado
y un baseline ingenuo, y elige el mejor modelo según una métrica principal.

1. Predice el test con cada modelo usando sus variables elegidas.
2. Calcula R², MAE, RMSE, MAPE, WAPE, sesgo, tasa de sobreestimación, MAE asimétrico,
   MAPE asimétrico y WAPE asimétrico.
3. Calcula las mismas métricas para el baseline ingenuo (flujo de hace 7 días).
4. Ordena los modelos por la métrica principal y entrega el mejor.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model_train import TrainResult

BASELINE_NAME = "baseline_lag_7"
BASELINE_COL = "net_cashflow_lag_7"
HIGHER_IS_BETTER = {"r2"}


@dataclass(frozen=True)
class EvalConfig:
    """Parámetros de la evaluación."""

    primary_metric: str = "asym_wape"
    mape_min_abs: float = 100_000.0
    over_weight: float = 2.0


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    mape_min_abs: float = 100_000.0,
    over_weight: float = 2.0,
) -> dict:
    """Calcula R², MAE, RMSE, MAPE, WAPE, sesgo, tasa de sobreestimación y MAE asimétrico.

    MAPE considera solo filas con |y| >= `mape_min_abs`; `mape_coverage` es el
    porcentaje de filas incluidas. `asym_mape` es el MAPE con los errores de
    sobreestimación multiplicados por `over_weight`. WAPE = Σ|error| / Σ|y|. Sesgo = media de
    (predicción - real); positivo indica sobreestimación. `over_rate` es el
    porcentaje de filas con predicción mayor al real. `asym_mae` es el MAE con
    los errores de sobreestimación multiplicados por `over_weight`. `asym_wape` =
    Σ(error asimétrico) / Σ|y|.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    mask = np.abs(y_true) >= mape_min_abs
    asym_error = np.where(error > 0, over_weight * error, -error)

    if mask.any():
        ape = np.abs(error[mask] / y_true[mask]) * 100
        weights = np.where(error[mask] > 0, over_weight, 1.0)
        mape = ape.mean()
        asym_mape = (weights * ape).mean()
    else:
        mape = asym_mape = np.nan

    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mape,
        "asym_mape": asym_mape,
        "mape_coverage": mask.mean() * 100,
        "wape": np.abs(error).sum() / np.abs(y_true).sum() * 100,
        "asym_wape": asym_error.sum() / np.abs(y_true).sum() * 100,
        "bias": error.mean(),
        "over_rate": (error > 0).mean() * 100,
        "asym_mae": asym_error.mean(),
    }


def baseline_predictions(test: pd.DataFrame, target: str, fallback: float) -> pd.Series:
    """Predicción ingenua: flujo del mismo comercio 7 días antes."""
    return test[BASELINE_COL].fillna(fallback)


def evaluate_models(
    results: dict[str, TrainResult],
    test: pd.DataFrame,
    target: str,
    baseline_fallback: float,
    config: EvalConfig = EvalConfig(),
) -> pd.DataFrame:
    """Entrega una tabla de métricas de test por modelo, incluido el baseline."""
    rows = []
    for name, result in results.items():
        y_pred = result.model.predict(test[result.features])
        metrics = regression_metrics(test[target], y_pred, config.mape_min_abs, config.over_weight)
        rows.append({"model": name, "n_features": len(result.features), **metrics})

    y_base = baseline_predictions(test, target, baseline_fallback)
    base_metrics = regression_metrics(test[target], y_base, config.mape_min_abs, config.over_weight)
    rows.append({"model": BASELINE_NAME, "n_features": 1, **base_metrics})

    table = pd.DataFrame(rows).set_index("model")
    ascending = config.primary_metric not in HIGHER_IS_BETTER
    return table.sort_values(config.primary_metric, ascending=ascending)


def select_best_model(table: pd.DataFrame, config: EvalConfig = EvalConfig()) -> str:
    """Entrega el nombre del mejor modelo según la métrica principal, excluido el baseline."""
    candidates = table.drop(index=BASELINE_NAME, errors="ignore")
    metric = candidates[config.primary_metric]
    if config.primary_metric in HIGHER_IS_BETTER:
        return metric.idxmax()
    return metric.idxmin()


if __name__ == "__main__":
    from src.data_analysis import select_features
    from src.data_loader import load_data
    from src.feature_eng import (
        TARGET,
        build_features,
        encode_categoricals,
        get_feature_columns,
    )
    from src.model_train import DATE_COL, TrainConfig, train_all

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:,.4f}".format)

    data = load_data()
    feats = build_features(data)
    test_cutoff = feats[DATE_COL].max() - pd.DateOffset(months=3)
    train_df = feats[feats[DATE_COL] <= test_cutoff].copy()
    test_df = feats[feats[DATE_COL] > test_cutoff].copy()
    train_df, encoders = encode_categoricals(train_df)
    test_df, _ = encode_categoricals(test_df, encoders)

    finalists = select_features(train_df, get_feature_columns(train_df), TARGET)["final_features"]
    fast_config = TrainConfig(n_trials=1, n_repeats=3, select_best_iteration=False)
    trained = train_all(train_df, finalists, TARGET, fast_config)

    eval_table = evaluate_models(trained, test_df, TARGET, train_df[TARGET].median())
    print(eval_table)
    print("mejor modelo:", select_best_model(eval_table))
