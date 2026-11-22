# Changelog

## 2026-11-22 (pipeline de selección)
- `config/train.yaml`: parámetros del pipeline; `main.py --config config/train.yaml` los lee.
- `data_cleaning.exclude_columns`: excluye `merchant_id` de las covariables; ya no se codifica `merchant_id_enc`.
- Métrica principal `asym_wape` (WAPE con sobreestimación ponderada x2) en `model_eval.py`.
- `n_trials=3`. Resultado en test: xgboost gana (asym_wape 47,5; WAPE 32,0; MAE 586.439) frente a RF (55,0), baseline lag 7 (80,1) y lineal (86,2).
- Lint corregido con ruff; `PyYAML` agregado a `requirements.txt`.

## 2026-11-22
- Parte 3: servicio FastAPI (`src/api.py`) con `GET /health` y `POST /predict`, validación pydantic y carga única del modelo.
- `src/feature_eng.py`: `build_features` parametrizable y `build_serving_features` para el historial de 14 días.
- `src/export_model.py`: entrena el modelo de servicio y guarda `models/model.joblib`.
- Dockerfile, `requirements-api.txt`, `examples/request.json` y tests (`tests/test_api.py`, `tests/test_features.py`).
- Pendiente: build de la imagen no verificado (Docker no está instalado en este equipo).
