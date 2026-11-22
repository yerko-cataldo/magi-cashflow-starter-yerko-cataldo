# Arquitectura

## Servicio de proyección (Parte 3)

```
request JSON ──> pydantic (PredictRequest) ──> build_serving_features ──> modelo ──> response
                                   ▲                    ▲
                           src/api.py          src/feature_eng.py (compartido con el entrenamiento)
```

- `src/export_model.py`: entrena y guarda `models/model.joblib` (modelo, columnas, codificadores, versión, métricas).
- `src/api.py`: carga el archivo al iniciar y expone `GET /health` y `POST /predict`.
- `src/feature_eng.py`: construcción de covariables. El servicio y el entrenamiento llaman a `build_serving_features`.
- `Dockerfile`: copia `api.py`, `feature_eng.py` y el artefacto `models/model.joblib` a la imagen.

## Decisiones de diseño

- **Información disponible al proyectar:** el request trae 14 días de `net_cashflow` y `txn_count`, más `is_holiday` y `opening_balance` del día. El modelo de servicio usa solo esas fuentes.
- **Sin `merchant_id` como variable:** se excluye con `data_cleaning.exclude_columns` en `main.py` y no se codifica en el modelo de servicio; los comercios nuevos reciben proyección con su historial.
- **Contrato:** se mantiene el del briefing. El historial exige exactamente 14 días consecutivos terminados el día previo a `forecast_date`.
- **Artefacto en la imagen:** el modelo se copia dentro de la imagen; cada versión del modelo implica una imagen nueva.
- **Carga única:** el modelo se carga en el `lifespan` de FastAPI; si el archivo no existe o está corrupto, el servicio no inicia.
