# Detalles técnicos

## Modelo exportado

- `RandomForestRegressor`: `n_estimators=150`, `max_depth=8`, `min_samples_leaf=20`, `max_features=0.5`, `random_state=67`.
- Objetivo: `net_cashflow` del día `t`.
- Covariables (55): calendario, `is_holiday`, `opening_balance`, rezagos 1/7/14 y estadísticos móviles de 7 y 14 días de `net_cashflow` y `txn_count` (calculados con datos hasta `t-1`), `merchant_category` y `region` codificadas.
- Evaluación: últimos 3 meses como test. Métrica principal de selección: WAPE asimétrico (`asym_wape`, `evaluation.primary_metric` en `config/train.yaml`), con `over_weight` aplicado a la sobreestimación. MAE modelo 607.455 vs baseline lag 7 980.201. Luego se reentrena con todo el histórico.

## API

| Endpoint | Respuesta |
|---|---|
| `GET /health` | `status`, `model_version`, `trained_until` |
| `POST /predict` | `merchant_id`, `forecast_date`, `predicted_net_cashflow`, `model_version` |

Validaciones (HTTP 422): campos obligatorios y tipos, historial de 14 días, fechas consecutivas terminadas en `forecast_date - 1`, `txn_count >= 0`, categoría y región conocidas por el modelo, covariables calculables.

## Dependencias

Servicio: `requirements-api.txt`. Desarrollo y tests: `requirements.txt`. Imagen base `python:3.14-slim`.

## Comandos

```bash
python -m src.export_model --version 1        # entrena y guarda models/model.joblib
python main.py --config config/train.yaml  # selección y evaluación de modelos
pytest                                        # tests
uvicorn src.api:app --port 8000               # servicio local
docker build -t magi-cashflow .
docker run -p 8000:8000 magi-cashflow
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/request.json
```

## Pipeline de entrenamiento y selección (`main.py`)

- Parámetros en `config/train.yaml`: separación temporal, outliers, selección de variables, modelos, `n_trials`, métrica principal.
- Modelos: regresión lineal, Random Forest y XGBoost con Optuna (`n_trials=3`, semilla 67) y eliminación escalonada de variables por importancia de permutación.
- Variables excluidas por `data_cleaning.exclude_columns`: `merchant_id`. `merchant_category` y `region` se codifican.
- Selección del mejor modelo: `asym_wape` en test (sobreestimación ponderada por `over_weight=2`). Optuna minimiza MAE de validación.
- Este pipeline usa el historial completo (rezagos hasta 30 y ventanas hasta 90 días). El modelo de servicio (`src/export_model.py`) usa solo las covariables disponibles con 14 días de historial.
