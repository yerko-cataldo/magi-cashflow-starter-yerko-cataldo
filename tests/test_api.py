"""Pruebas del servicio FastAPI."""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

EXAMPLE = json.loads((Path(__file__).resolve().parent.parent / "examples" / "request.json").read_text())


@pytest.fixture()
def client(model_file, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", model_file)
    with TestClient(app) as c:
        yield c


def valid_payload() -> dict:
    payload = copy.deepcopy(EXAMPLE)
    payload["merchant_category"] = "retail"
    payload["region"] = "Sur"
    return payload


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test"


def test_predict_ok(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["merchant_id"] == EXAMPLE["merchant_id"]
    assert body["forecast_date"] == EXAMPLE["forecast_date"]
    assert isinstance(body["predicted_net_cashflow"], float)
    assert body["model_version"] == "test"


def test_predict_is_deterministic(client):
    first = client.post("/predict", json=valid_payload()).json()
    second = client.post("/predict", json=valid_payload()).json()
    assert first == second


def test_short_history_rejected(client):
    payload = valid_payload()
    payload["history"] = payload["history"][:-1]
    assert client.post("/predict", json=payload).status_code == 422


def test_date_gap_rejected(client):
    payload = valid_payload()
    payload["history"][5]["date"] = "2020-01-01"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "consecutivos" in response.text


def test_history_must_end_day_before_forecast(client):
    payload = valid_payload()
    payload["forecast_date"] = "2030-01-01"
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_field_rejected(client):
    payload = valid_payload()
    del payload["opening_balance"]
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_category_rejected(client):
    payload = valid_payload()
    payload["merchant_category"] = "inexistente"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Valores aceptados" in response.text
