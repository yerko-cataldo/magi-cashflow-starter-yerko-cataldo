"""Entrenamiento de modelos.

Entrena regresión lineal (modelo base), Random Forest y XGBoost con búsqueda de
hiperparámetros en Optuna y eliminación escalonada de variables por importancia.

1. Separa el conjunto de entrenamiento en ajuste y validación por fecha.
2. Por modelo, busca hiperparámetros con Optuna minimizando el MAE de validación.
3. Calcula la importancia por permutación de cada variable sobre validación,
   expresada como proporción del total.
4. Elimina la variable de menor importancia si está bajo `min_importance`.
   Las variables sobre `alert_importance` se registran como alerta y no se eliminan.
5. Repite 2 a 4 hasta que todas las variables superen `min_importance`.
6. Elige la iteración con menor MAE de validación (a igual MAE, la de menos
   variables) y reentrena con todo el train.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_analysis import CATEGORICAL_ENC_COLS

SEED = 67
DATE_COL = "date"


@dataclass(frozen=True)
class TrainConfig:
    """Parámetros del entrenamiento y de la eliminación de variables."""

    n_trials: int = 5
    valid_months: int = 2
    min_importance: float = 0.01
    alert_importance: float = 0.40
    min_features: int = 1
    max_iterations: int = 50
    n_repeats: int = 5
    select_best_iteration: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """Definición de un modelo: espacio de búsqueda y constructor."""

    name: str
    suggest: Callable[[optuna.Trial], dict]
    build: Callable[[dict, list[str]], object]
    tunable: bool = True


@dataclass
class TrainResult:
    """Resultado del entrenamiento de un modelo."""

    name: str
    model: object
    features: list[str]
    params: dict
    valid_mae: float
    history: list[dict] = field(default_factory=list)


def set_seeds(seed: int = SEED) -> None:
    """Fija las semillas de numpy y random."""
    np.random.seed(seed)
    random.seed(seed)


def _linear_model(estimator, features: list[str]) -> TransformedTargetRegressor:
    """Pipeline lineal: imputación, escalamiento, one-hot y objetivo estandarizado."""
    categorical = [c for c in features if c in CATEGORICAL_ENC_COLS]
    numeric = [c for c in features if c not in categorical]

    transformers = []
    if numeric:
        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", onehot, categorical))

    pipeline = Pipeline(
        [
            ("prep", ColumnTransformer(transformers)),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def _suggest_linear(trial: optuna.Trial) -> dict:
    return {}


def _build_linear(params: dict, features: list[str]):
    return _linear_model(LinearRegression(), features)


def _suggest_rf(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50, log=True),
        "max_features": trial.suggest_float("max_features", 0.3, 0.7),
    }


def _build_rf(params: dict, features: list[str]):
    return RandomForestRegressor(**params, n_jobs=-1, random_state=SEED)


def _suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _build_xgb(params: dict, features: list[str]):
    return XGBRegressor(**params, tree_method="hist", n_jobs=-1, random_state=SEED)


MODEL_SPECS = {
    "linear": ModelSpec("linear", _suggest_linear, _build_linear, tunable=False),
    "random_forest": ModelSpec("random_forest", _suggest_rf, _build_rf),
    "xgboost": ModelSpec("xgboost", _suggest_xgb, _build_xgb),
}


def temporal_split(df: pd.DataFrame, valid_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa en ajuste y validación; validación son los últimos `valid_months` meses."""
    cutoff = df[DATE_COL].max() - pd.DateOffset(months=valid_months)
    return df[df[DATE_COL] <= cutoff], df[df[DATE_COL] > cutoff]


def tune(
    spec: ModelSpec,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int,
    warm_params: dict | None = None,
) -> tuple[dict, float, object]:
    """Busca hiperparámetros con Optuna y entrega parámetros, MAE y modelo del mejor trial."""
    features = list(X_fit.columns)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if warm_params is not None:
        study.enqueue_trial(warm_params)

    best = {"mae": np.inf, "model": None}

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest(trial)
        model = spec.build(params, features)
        model.fit(X_fit, y_fit)
        mae = mean_absolute_error(y_val, model.predict(X_val))
        if mae < best["mae"]:
            best["mae"] = mae
            best["model"] = model
        return mae

    study.optimize(objective, n_trials=n_trials if spec.tunable else 1)
    return study.best_params, study.best_value, best["model"]


def importance_shares(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int,
) -> pd.Series:
    """Importancia por permutación sobre MAE, como proporción del total."""
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=SEED,
    )
    importance = pd.Series(result.importances_mean, index=X_val.columns).clip(lower=0)
    total = importance.sum()
    if total == 0:
        return importance
    return importance / total


def pick_feature_to_drop(shares: pd.Series, min_importance: float) -> str | None:
    """Entrega la variable de menor importancia bajo el mínimo, o None si no hay."""
    low = shares[shares < min_importance]
    if low.empty:
        return None
    return low.idxmin()


def train_model(
    spec: ModelSpec,
    train: pd.DataFrame,
    features: list[str],
    target: str,
    config: TrainConfig = TrainConfig(),
) -> TrainResult:
    """Entrena un modelo con búsqueda de hiperparámetros y eliminación escalonada de variables."""
    fit_df, val_df = temporal_split(train, config.valid_months)
    current = list(features)
    history = []
    warm_params = None

    for iteration in range(config.max_iterations):
        params, valid_mae, model = tune(
            spec,
            fit_df[current],
            fit_df[target],
            val_df[current],
            val_df[target],
            config.n_trials,
            warm_params,
        )
        shares = importance_shares(model, val_df[current], val_df[target], config.n_repeats)
        to_drop = None
        if len(current) > config.min_features:
            to_drop = pick_feature_to_drop(shares, config.min_importance)
        alerts = shares[shares > config.alert_importance].index.tolist()

        history.append(
            {
                "iteration": iteration,
                "n_features": len(current),
                "features": list(current),
                "params": params,
                "valid_mae": valid_mae,
                "importance": shares.sort_values(ascending=False),
                "dropped": to_drop,
                "alerts": alerts,
            }
        )
        if to_drop is None:
            break
        current.remove(to_drop)
        warm_params = params

    if config.select_best_iteration:
        chosen = min(history, key=lambda h: (round(h["valid_mae"]), h["n_features"]))
    else:
        chosen = history[-1]

    final_model = spec.build(chosen["params"], chosen["features"])
    final_model.fit(train[chosen["features"]], train[target])

    return TrainResult(
        name=spec.name,
        model=final_model,
        features=chosen["features"],
        params=chosen["params"],
        valid_mae=chosen["valid_mae"],
        history=history,
    )


def train_all(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    config: TrainConfig = TrainConfig(),
    model_names: list[str] | None = None,
) -> dict[str, TrainResult]:
    """Entrena todos los modelos indicados y entrega sus resultados por nombre."""
    set_seeds()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    names = model_names or list(MODEL_SPECS)
    return {name: train_model(MODEL_SPECS[name], train, features, target, config) for name in names}


def history_table(result: TrainResult) -> pd.DataFrame:
    """Resume la evolución de iteraciones de un modelo."""
    rows = [
        {
            "iteration": h["iteration"],
            "n_features": h["n_features"],
            "valid_mae": h["valid_mae"],
            "top_feature": h["importance"].index[0],
            "top_share": h["importance"].iloc[0],
            "dropped": h["dropped"],
            "alerts": ", ".join(h["alerts"]),
        }
        for h in result.history
    ]
    return pd.DataFrame(rows).set_index("iteration")


if __name__ == "__main__":
    import time

    from src.data_analysis import select_features
    from src.data_loader import load_data
    from src.feature_eng import (
        TARGET,
        build_features,
        encode_categoricals,
        get_feature_columns,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    data = load_data()
    feats = build_features(data)
    test_cutoff = feats[DATE_COL].max() - pd.DateOffset(months=3)
    train_df = feats[feats[DATE_COL] <= test_cutoff].copy()
    train_df, _ = encode_categoricals(train_df)

    selection = select_features(train_df, get_feature_columns(train_df), TARGET)
    finalists = selection["final_features"]
    print("variables finalistas:", len(finalists))

    baseline_val = temporal_split(train_df, TrainConfig().valid_months)[1]
    baseline_mae = mean_absolute_error(
        baseline_val[TARGET], baseline_val["net_cashflow_lag_7"].fillna(baseline_val[TARGET].median())
    )
    print(f"baseline ingenuo (lag 7) MAE validación: {baseline_mae:,.0f}")

    for model_name in MODEL_SPECS:
        start = time.time()
        res = train_all(train_df, finalists, TARGET, model_names=[model_name])[model_name]
        print()
        print(f"== {model_name} ({time.time() - start:.0f}s) ==")
        print(history_table(res).round(4))
        print(f"elegido: {len(res.features)} variables, MAE validación {res.valid_mae:,.0f}")
        print("parámetros:", res.params)
