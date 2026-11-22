"""Pipeline de proyección de flujo de caja.

Ejecuta el flujo completo, de los datos crudos al mejor modelo. Los parámetros se
leen de `config/train.yaml`.

1. Carga los datos y elimina duplicados.
2. Separa train y test por fecha.
3. Acota outliers con límites calculados en train (opcional).
4. Crea covariables, excluye columnas configuradas y codifica categóricas.
5. Selecciona variables: univariado, poder predictivo (η²) y colinealidad.
6. Entrena los modelos con Optuna y eliminación escalonada de variables.
7. Evalúa en test y elige el mejor modelo.

Ejecución: python main.py --config config/train.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data_analysis import AnalysisConfig, select_features
from src.data_cleaning import (
    OutlierConfig,
    cap_outliers,
    exclude_columns,
    remove_duplicates,
)
from src.data_loader import load_data
from src.feature_eng import (
    TARGET,
    build_features,
    encode_categoricals,
    get_feature_columns,
)
from src.model_eval import EvalConfig, evaluate_models, select_best_model
from src.model_train import TrainConfig, history_table, train_all

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train.yaml"


def load_config(path: Path) -> dict:
    """Lee los parámetros del pipeline desde un archivo YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: Path = DEFAULT_CONFIG) -> None:
    cfg = load_config(config_path)
    split_cfg = cfg["split"]
    outliers = cfg["outliers"]
    selection_cfg = cfg["selection"]
    training = cfg["training"]
    evaluation = cfg["evaluation"]
    excluded = cfg["features"]["exclude_columns"]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:,.2f}".format)

    # 1. Carga y duplicados
    data = load_data()
    data, dup_report = remove_duplicates(data)
    print("1. filas:", dup_report["n_final"], "| duplicados eliminados:",
          dup_report["n_exact_duplicates"] + dup_report["n_key_duplicates"])

    # 2. Split temporal
    test_cutoff = data["date"].max() - pd.DateOffset(months=split_cfg["test_months"])
    train_raw = data[data["date"] <= test_cutoff]
    test_raw = data[data["date"] > test_cutoff]
    print(f"2. train hasta {test_cutoff.date()}: {len(train_raw)} filas | test: {len(test_raw)} filas")

    # 3. Outliers
    outlier_cfg = OutlierConfig(enabled=outliers["enabled"], columns=outliers["columns"], k=outliers["iqr_k"])
    train_raw, test_raw, _ = cap_outliers(train_raw, test_raw, outlier_cfg)
    print("3. outliers acotados:", outliers["enabled"])

    # 4. Covariables
    full = pd.concat([train_raw, test_raw])
    features_df = build_features(full)
    features_df = exclude_columns(features_df, excluded)
    train = features_df[features_df["date"] <= test_cutoff].copy()
    test = features_df[features_df["date"] > test_cutoff].copy()
    train, encoders = encode_categoricals(train)
    test, _ = encode_categoricals(test, encoders)
    candidates = get_feature_columns(train)
    print("4. covariables candidatas:", len(candidates), "| columnas excluidas:", excluded)

    # 5. Selección de variables
    analysis_cfg = AnalysisConfig(
        max_null_pct=selection_cfg["max_null_pct"],
        min_power=selection_cfg["min_power"],
        max_corr=selection_cfg["max_corr"],
    )
    selection = select_features(train, candidates, TARGET, analysis_cfg)
    finalists = selection["final_features"]
    print(f"5. {selection['n_initial']} -> univariado {selection['n_after_univariate']}"
          f" -> poder predictivo {selection['n_after_bivariate']} -> colinealidad {len(finalists)}")

    # 6. Entrenamiento
    train_cfg = TrainConfig(
        n_trials=training["n_trials"],
        valid_months=split_cfg["valid_months"],
        min_importance=training["min_importance"],
        alert_importance=training["alert_importance"],
    )
    trained = train_all(train, finalists, TARGET, train_cfg, training["models"])
    for name, result in trained.items():
        print(f"\n6. {name}: {len(result.features)} variables, MAE validación {result.valid_mae:,.0f}")
        print(history_table(result)[["n_features", "valid_mae", "top_feature", "top_share", "dropped"]])

    # 7. Evaluación
    eval_cfg = EvalConfig(
        primary_metric=evaluation["primary_metric"],
        mape_min_abs=evaluation["mape_min_abs"],
        over_weight=evaluation["over_weight"],
    )
    table = evaluate_models(trained, test, TARGET, train[TARGET].median(), eval_cfg)
    best = select_best_model(table, eval_cfg)
    print(f"\n7. métricas en test (orden: {eval_cfg.primary_metric})")
    print(table)
    print("\nmejor modelo:", best)
    print("variables:", trained[best].features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    main(args.config)
