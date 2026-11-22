

## Feature Engineering


[CRÍTICA] No challenge de modelos

Se aplica solamente un random forest. Sin comparar contra otros modelos más parsimoniosos como regresiones, o modelos más complejos como XGBoost o redes neuronales, pudiendo perder explicabilidad o precisión.


[CRÍTICA] Validación cruzada 

La separación en data de entrenamiento y de validación es aleatoria con todo el dataset. En casos de datos temporales, la separación debe ser temporal: dejar por ejemplo los últimos 3 meses como data de validación, para que no exista data leakage y el modelo se prepare mejor ante los casos reales futuros.


[CRÍTICA] Importancia de variables desbalanceada

Solo una variable concentra más del 98% del peso...

La variable más pesada no debería superar el 30-40% para un modelo estable y balanceado.

Tomando este caso, el modelo solo está prediciendo en función de la variación diaria en el saldo, ignorando todos los otros factores. Produciendo clarísimos sesgos.

Desde la tercera variable en adelante en cuanto a importancia, aportan nula información al modelo. Una variable no se debería considerar en el modelo final si obtiene importancia menor al 1%.


[MEDIA] Pocas variables creadas. 

Es sumamente importante realizar un buen feature engineering para lograr capturar toda la información de los datos.

En el notebook entregado solo se consideran:

Lags a 1 día
Lags a 7 días
Media móvil a 7 días
Media móvil a 30 días
Desviación estándar móvil a 7 días
Variación diaria de saldo
Ticket promedio

Podrían agregarse variables de como mínimo, máximo, mediana, coeviciente de variación, entre otras, y distintas interacciones de net_cashflow (min/max, max/media, etc.) o entre otras covariables (inflows/outflows, inflows/txn_count, etc.) y aumentar el largo de las ventanas a 1, 7, 14, 30, 60, 90. Este dataset solo tiene un año y medio de historia, con más datos podríamos considerar 365 días también.

También pueden existir otras variables interesantes, como el tiempo hasta el mínimo/máximo cashflow, la tendencia del último tiempo, etc.

No hay problema con agregar muchas variables, ya que posteriormente se pueden filtrar por completitud, poder predictivo, colinealidad o importancia para el modelo final.


[BAJA] No análisis de outliers 

No se realiza un análisis de valores extremos, ni se explica la decisión sobre esto.


[MEDIO] No análisis de poder predictivo o colinealidad 

No se realiza un análisis de poder predictivo de las distintas covariables, ya sea por Information Value o transformación a WOEs.

Tampoco se revisa si entre las variables originales o las creadas existe redundancia en la información, ya sea cortando según correlación inter-variables, o con algún método de reducción como PCA.


[MEDIO] Pocas métricas de rendimiento

Solo se evalúa el R2 y el MAPE, sin considerar otras métricas como RMSE o el MAE. Y tampoco se habla de la prioridad de disminuir el error de predicción hacia arriba o hacia abajo.

Al solo evaluar el error absoluto porcentual medio, puede que exista diferencia en los resultados de esta métrica dependiendo del volumen del valor a predecir. 

Por eso siempre es importante mirar múltiples métricas en la evaluación de un modelo en contexto de regresión.


[BAJA] Conclusiones 

Resultados tan altos deberían ser una alarma.

Esas no son las variables más importantes, sino la diferencia diaria entre el saldo de la cuenta, que es exactamente lo que se busca predecir...




