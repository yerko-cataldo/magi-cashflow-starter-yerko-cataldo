FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/api.py src/feature_eng.py src/
COPY models/model.joblib models/model.joblib

ENV MODEL_PATH=/app/models/model.joblib

RUN useradd --no-create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
