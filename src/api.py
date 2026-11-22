"""Servicio REST de proyección de flujo de caja.

Expone el modelo exportado con FastAPI. Calcula las covariables con la misma
función que el entrenamiento (`build_serving_features`).

1. Al iniciar, carga el modelo una sola vez desde `MODEL_PATH`.
2. `GET /health` entrega el estado y la versión del modelo.
3. `POST /predict` valida la entrada, construye las covariables y entrega la proyección.

Variables de entorno: MODEL_PATH (ruta del archivo del modelo).
"""

import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from src.feature_eng import (
    HISTORY_DAYS,
    SERVING_CATEGORICALS,
    build_serving_features,
    encode_categoricals,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


class HistoryDay(BaseModel):
    """Flujo y transacciones de un día del historial."""

    date: date
    net_cashflow: float
    txn_count: int = Field(ge=0)


class PredictRequest(BaseModel):
    """Datos del comercio y su historial reciente."""

    merchant_id: str = Field(min_length=1)
    forecast_date: date
    merchant_category: str = Field(min_length=1)
    region: str = Field(min_length=1)
    is_holiday: bool
    opening_balance: float
    history: list[HistoryDay] = Field(min_length=HISTORY_DAYS, max_length=HISTORY_DAYS)

    @model_validator(mode="after")
    def check_history_dates(self) -> "PredictRequest":
        """Exige fechas consecutivas, ascendentes y terminadas el día previo a `forecast_date`."""
        dates = [h.date for h in self.history]
        expected = [self.forecast_date - timedelta(days=HISTORY_DAYS - i) for i in range(HISTORY_DAYS)]
        if dates != expected:
            raise ValueError(
                f"history debe tener {HISTORY_DAYS} días consecutivos y ascendentes, "
                f"desde {expected[0]} hasta {expected[-1]} (día previo a forecast_date)"
            )
        return self


class PredictResponse(BaseModel):
    """Proyección de flujo neto."""

    merchant_id: str
    forecast_date: date
    predicted_net_cashflow: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar el servicio."""
    path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.bundle = joblib.load(path)
    yield


app = FastAPI(title="MAGI Cashflow", lifespan=lifespan)


def build_input_row(req: PredictRequest, bundle: dict) -> pd.DataFrame:
    """Arma el historial más la fila a proyectar y entrega las covariables de esa fila."""
    rows = [
        {"date": pd.Timestamp(h.date), "net_cashflow": h.net_cashflow, "txn_count": h.txn_count}
        for h in req.history
    ]
    rows.append({"date": pd.Timestamp(req.forecast_date), "net_cashflow": None, "txn_count": None})
    df = pd.DataFrame(rows)
    df["merchant_id"] = req.merchant_id
    df["merchant_category"] = req.merchant_category
    df["region"] = req.region
    df["is_holiday"] = int(req.is_holiday)
    df["opening_balance"] = req.opening_balance

    df = build_serving_features(df)
    df, _ = encode_categoricals(df, bundle["encoders"], SERVING_CATEGORICALS)
    return df.tail(1)[bundle["columns"]]


@app.get("/health")
def health(request: Request) -> dict:
    """Entrega el estado del servicio y la versión del modelo cargado."""
    bundle = request.app.state.bundle
    return {
        "status": "ok",
        "model_version": bundle["version"],
        "trained_until": bundle["trained_until"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """Entrega la proyección de flujo neto para `forecast_date`."""
    bundle = request.app.state.bundle

    for col in SERVING_CATEGORICALS:
        known = list(bundle["encoders"][col].classes_)
        value = getattr(req, col)
        if value not in known:
            raise HTTPException(status_code=422, detail=f"{col} '{value}' no es válido. Valores aceptados: {known}")

    row = build_input_row(req, bundle)
    if row.isna().any(axis=None):
        raise HTTPException(status_code=422, detail="El historial no permite calcular las covariables.")

    prediction = float(bundle["model"].predict(row)[0])
    return PredictResponse(
        merchant_id=req.merchant_id,
        forecast_date=req.forecast_date,
        predicted_net_cashflow=prediction,
        model_version=bundle["version"],
    )
