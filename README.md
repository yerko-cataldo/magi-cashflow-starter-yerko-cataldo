# Modelo de Proyección de Flujo de Caja

**Proveedor:** SEELE Analytics
**Cliente:** NERV Financial Division — Tesorería

Proyecta el flujo de caja neto diario (`net_cashflow`) de cada comercio. Incluye el pipeline de entrenamiento y selección de modelos (`main.py`) y un servicio HTTP (FastAPI) que sirve el modelo entrenado.

## Overview y arquitectura

```
data/daily_cashflow.csv
        │
        ▼
main.py  ──► selección de variables, entrenamiento (lineal, RF, XGBoost) y evaluación en test
        │
src/export_model.py  ──► entrena el modelo de servicio y guarda models/model.joblib
        │
        ▼
request JSON ──► pydantic (PredictRequest) ──► build_serving_features ──► modelo ──► response
                                  ▲                    ▲
                          src/api.py          src/feature_eng.py (compartido con el entrenamiento)
```

- `src/data_loader.py`, `src/data_cleaning.py`, `src/data_analysis.py`: carga, limpieza y selección de variables.
- `src/feature_eng.py`: construcción de covariables. El entrenamiento y el servicio usan las mismas funciones.
- `src/model_train.py`, `src/model_eval.py`: entrenamiento con Optuna y evaluación en test.
- `src/export_model.py`: genera `models/model.joblib` (modelo, columnas, codificadores, versión, métricas).
- `src/api.py`: carga el artefacto al iniciar y expone `GET /health` y `POST /predict`.
- `Dockerfile`: empaqueta el servicio y el artefacto en una imagen `python:3.14-slim`.

Decisiones principales (detalle en `docs/architecture.md`):

- El servicio usa solo información disponible al proyectar: 14 días de `net_cashflow` y `txn_count`, más `is_holiday` y `opening_balance` del día.
- `merchant_id` no es variable. Se excluye, y los comercios nuevos reciben proyección con su historial.
- El modelo se carga una vez en el `lifespan` de FastAPI. Si el archivo no existe o está corrupto, el servicio no inicia.

## Instalación

Requisitos: Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt  # entrenamiento y tests
pip install -r requirements-api.txt  # solo servicio (opcional, para la imagen)
```

## Comandos

```bash
# Entrenar y guardar el modelo de servicio (models/model.joblib)
python -m src.export_model --version 1

# Pipeline completo de selección y evaluación de modelos
python main.py --config config/train.yaml

# Tests
pytest

# Levantar el servicio localmente
uvicorn src.api:app --port 8000

# Docker
docker build -t magi-cashflow .
docker run -p 8000:8000 magi-cashflow
```

## Estrategia de validación y métricas

**Validación temporal.** No se usa una separación aleatoria, porque los datos son una serie de tiempo y una partición aleatoria filtraría información del futuro. Se separa así:

- Test: últimos 3 meses (`split.test_months: 3`).
- Validación: 2 meses antes del test (`split.valid_months: 2`), usada por Optuna.
- Reentrenamiento final con todo el histórico después de elegir el modelo.

**Métrica principal: WAPE asimétrico (`asym_wape`).** Para Tesorería el error no cuesta lo mismo en ambos sentidos. Sobreestimar la caja es más caro que subestimarla, así que la sobreestimación pesa el doble (`over_weight: 2.0`).

**Métricas de apoyo:** MAE, RMSE y R². El MAPE se reporta, pero no decide la selección porque se distorsiona con valores cercanos a cero (`mape_min_abs: 100000`).

**Selección de variables:** filtro por nulos (>90%), poder predictivo (η² ≥ 0.01) y colinealidad (correlación > 0.6). Luego eliminación escalonada por importancia de permutación.

## Resultados

Los números de esta sección provienen de `docs/technical.md` y de las salidas del notebook original. En esta entrega **no se volvió a ejecutar** `main.py` para regenerarlos; conviene correrlo antes de presentarlos como definitivos.

| Modelo | MAE en test | Comentario |
|---|---|---|
| Baseline (lag 7) | 7 980 | Repite el valor de hace 7 días |
| Random Forest (modelo de servicio) | 607 | Métrica reportada en `docs/technical.md` |
| Notebook original (Random Forest) | R² 0.9998 | Reportado como "precisión 99.9%" |

Comparación con el notebook original: el notebook reporta R² ≈ 0.9998 y MAPE 0.09%. Estas cifras son muy altas para un problema de flujo de caja diario y, según la revisión en `REVIEW.md`, están probablemente infladas por la fuga de información (la variable dominante es la variación diaria del saldo, que es casi el mismo objetivo). Por eso el R² del notebook no se toma como referencia de calidad, y la comparación útil es contra el baseline.

## Ejemplo de request

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/request.json
```

Contenido resumido de `examples/request.json`:

```json
{
  "merchant_id": "M-0042",
  "forecast_date": "2026-11-01",
  "merchant_category": "electronica",
  "region": "Metropolitana",
  "is_holiday": false,
  "opening_balance": 407233048,
  "history": [
    {"date": "2026-10-18", "net_cashflow": 1090701, "txn_count": 28},
    {"date": "2026-10-19", "net_cashflow": -4412293, "txn_count": 24}
    // ... hasta 14 días consecutivos terminados en 2026-10-31
  ]
}
```

Respuesta esperada: `merchant_id`, `forecast_date`, `predicted_net_cashflow`, `model_version`.

Validaciones (HTTP 422): campos obligatorios y tipos, historial de exactamente 14 días consecutivos terminados el día anterior a `forecast_date`, `txn_count >= 0`, categoría y región conocidas por el modelo.

Endpoint de salud: `GET /health` devuelve `status`, `model_version` y `trained_until`.

## Limitaciones conocidas y próximas mejoras

- **Fuga de información en el notebook original.** El R² 0.9998 no es creíble. Hay que rehacer el análisis excluyendo la variación de saldo como predictor.
- **Importancia de variables desbalanceada.** Según `REVIEW.md`, una sola variable concentra más del 98% de la importancia. El umbral recomendado es 30–40%.
- **Pipeline y servicio usan información distinta.** `main.py` usa historial de hasta 90 días; el servicio solo 14 días. Las métricas de `main.py` no describen exactamente al servicio.
- **Búsqueda de hiperparámetros corta.** `n_trials: 3` en Optuna es insuficiente para una conclusión sólida.
- **Feature engineering acotado.** Faltan mínimos, máximos, medianas, coeficientes de variación, ventanas de 30/60/90 días y tendencias.
- **Docker no verificado.** La imagen no se ha construido en este equipo (Docker no está instalado).
- **Reentrenamiento.** Recomendado cada 6 meses; cada versión del modelo exige construir una imagen nueva porque el artefacto va dentro de la imagen.
- **Sin monitoreo de drift ni alertas** en producción (ver `MONITORING.md`).

## Costo para Tesorería

Para Tesorería el error del modelo no cuesta lo mismo en ambos sentidos. Sobreestimar la caja es más caro que subestimar, ya que es mejor tener un colchón de excedentes por subestimación, a quedarse con flujo faltante y tener que postergar pagos o cubrir con otros financiamientos.

Esto se aplicará en el modelo mediante una métrica de rendimiento asimétrica (`asym_wape`) que penalice el doble la sobreestimación.
