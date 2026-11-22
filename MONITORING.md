
### ¿Qué métricas monitorearías con el modelo en producción y con qué frecuencia?

Monitorearía diariamente la entrega de las predicciones en calidad, formato y tiempo.

Cada 1, 3 o 6 meses el model drift y data drift; cómo ha ido el rendimiento de las predicciones (MAPE o MAPE asimétrico) y cómo se han comportado las variables de entrada en cuanto a distribución y volúmenes.


### ¿Cómo detectarías drift de datos y drift de concepto? Menciona técnicas o herramientas concretas

Drift de datos:
Monitoreando KS o PSI oara verificar que las distribuciones de los datos no se hayan desviado de los datos con los que se entrenó el modelo.

Drift de concepto:
Monitoreando que el MAPE/MAPE asimétrico no aumente (que el rendimiento no empeore), para comprobar que las relaciones entre las covariables y el target no ha cambiado.

### El flujo de caja tiene estacionalidad fuerte (fines de mes, feriados, eventos comerciales). ¿Cómo distinguirías una degradación real del modelo de una variación estacional esperada?

Comparando el error estacionario del modelo (lag 30 dias, o comparando contra los festivos anteriores) y viendo si el error siempre sube en esas épocas, o si esta ves sube más, por ende sería degradación del modelo.

Sería necesario mantener una base de datos de feriados, cyberdays, y fechas especiales que podrían generar variaciones en el rendimiento de las predicciones orgánicamente.


### ¿Cómo tratarías a los comercios nuevos, que aún no tienen historial suficiente?

Se necesitaría un historial mínimo de datos dependiendo de las variables que alimentan las predicciones del modelo.

Si aún no existe datos, se puede realizar un ejercicio de segmentación para encontrar similaridad con otros comercios y usar esas predicciones para estimar. Marcando estas estimaciones como de baja confianza.


### ¿Qué condiciones gatillarían un re-entrenamiento y cuáles un rollback a la versión anterior?

Deterioro significativo o acumulado del MAPE, PSI alto, o muchos cambios en cuanto a nuevos/distintos comercios, justificarían un re-entrenamiento.

Si un nuevo despliegue genera errores en la entrega de predicciones (errores en la API) habría que hacer rollback.


### ¿Qué alertas debería recibir el equipo de Tesorería y cuáles el equipo técnico?

Tesorería: alertas de que las variables de entrada no estén capturando algún dato, para que puedan gestionar por su lado.
Alertas de que ciertos segmentos están obteniendo menor rendimiento, para que puedan entender el porqué.
Alertas de que las predicciones estén siendo de baja confianza, dado el aumento del WAPE.

Técnico: errores de la API o demasiada latencia, MAE sobre un umbral de alarma definido, o drift (PSI, KS).

