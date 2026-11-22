"""Exportación del modelo para servicio.

Entrena un Random Forest con las covariables disponibles al momento de proyectar y
guarda el modelo con su metadata en un único archivo.

1. Carga los datos y elimina duplicados.
2. Construye las covariables de servicio sobre todo el histórico.
3. Separa train y test por fecha y mide MAE contra el baseline de lag 7.
4. Reentrena con todo el histórico.
5. Guarda modelo, columnas, codificadores, versión y métricas en `models/model.joblib`.

Ejecución: python -m src.export_model --version 1
"""

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.data_cleaning import remove_duplicates
from src.data_loader import load_data
from src.feature_eng import (
    SERVING_CATEGORICALS,
    TARGET,
    build_serving_features,
    encode_categoricals,
    get_feature_columns,
)

SEED = 67
TEST_MONTHS = 3
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
RF_PARAMS = {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 20, "max_features": 0.5}


def model_columns(df: pd.DataFrame) -> list[str]:
    """Entrega las columnas de entrada del modelo (numéricas y categóricas codificadas)."""
    return get_feature_columns(df)


def main(version: str, path: Path = MODEL_PATH) -> None:
    np.random.seed(SEED)
    random.seed(SEED)

    data, _ = remove_duplicates(load_data())
    features_df = build_serving_features(data)
    features_df, encoders = encode_categoricals(features_df, columns=SERVING_CATEGORICALS)
    columns = model_columns(features_df)
    features_df = features_df.dropna(subset=columns)

    cutoff = features_df["date"].max() - pd.DateOffset(months=TEST_MONTHS)
    train = features_df[features_df["date"] <= cutoff]
    test = features_df[features_df["date"] > cutoff]

    holdout = RandomForestRegressor(**RF_PARAMS, n_jobs=-1, random_state=SEED)
    holdout.fit(train[columns], train[TARGET])
    mae_model = mean_absolute_error(test[TARGET], holdout.predict(test[columns]))
    mae_baseline = mean_absolute_error(test[TARGET], test["net_cashflow_lag_7"])
    print(f"MAE test: modelo {mae_model:,.0f} | baseline lag 7 {mae_baseline:,.0f}")

    final = RandomForestRegressor(**RF_PARAMS, n_jobs=-1, random_state=SEED)
    final.fit(features_df[columns], features_df[TARGET])

    bundle = {
        "model": final,
        "columns": columns,
        "encoders": encoders,
        "version": version,
        "metrics": {"test_mae": mae_model, "baseline_lag7_mae": mae_baseline},
        "trained_until": str(features_df["date"].max().date()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print("modelo guardado en", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    main(args.version, args.output)
