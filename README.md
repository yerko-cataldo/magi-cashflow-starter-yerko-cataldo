# Modelo de Proyección de Flujo de Caja — Entrega v0

**Proveedor:** SEELE Analytics
**Cliente:** NERV Financial Division — Tesorería

## Contenido

- `data/daily_cashflow.csv` — histórico diario de flujo de caja por comercio (mayo 2025 a octubre 2026).
- `notebooks/cashflow_model_v0.ipynb` — notebook con el desarrollo completo del modelo.

## Resumen

Se entrenó un modelo Random Forest que proyecta el flujo de caja neto diario (`net_cashflow`) de cada comercio. El modelo alcanza un R² superior a 0.99 en el conjunto de prueba, por lo que lo consideramos listo para ser desplegado en producción.

## Cómo ejecutar

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
cd notebooks
jupyter notebook cashflow_model_v0.ipynb
```

Ejecutar todas las celdas en orden. Al final se genera el archivo `model_final_v2.pkl` con el modelo entrenado.

## Notas

- Los resultados pueden variar levemente entre ejecuciones.
- Para usar el modelo en producción basta con cargar el `.pkl` y llamar a `model.predict()`.
- Se recomienda re-entrenar cada 6 meses.
