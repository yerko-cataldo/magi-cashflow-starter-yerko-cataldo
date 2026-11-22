"""Carga de datos de flujo de caja.

Lee el CSV diario de flujo de caja por comercio y entrega estadísticos descriptivos.

1. Lee el CSV en un DataFrame.
2. Convierte la columna `date` a datetime.
3. Ordena por `merchant_id` y `date`.
4. Calcula shape, media, desviación estándar, cuartiles y valores faltantes.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "daily_cashflow.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lee el CSV, convierte `date` a datetime y ordena por comercio y fecha."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.sort_values(["merchant_id", "date"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Entrega shape, estadísticos de columnas numéricas y valores faltantes."""
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    missing = pd.DataFrame(
        {
            "n_missing": df.isna().sum(),
            "pct_missing": df.isna().mean() * 100,
        }
    )
    return {"shape": df.shape, "stats": stats, "missing": missing}


if __name__ == "__main__":
    data = load_data()
    summary = summarize(data)
    print("shape:", summary["shape"])
    print(summary["stats"])
    print(summary["missing"])
