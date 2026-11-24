# Resultados de la versión cruda (v0)

Corrida del 23/11/2026 en Colab (Tesla T4), solo la partición 0 de 5, con el script crudo y dos cambios de ejecución:
`USAR_DRIVE = False` y `PARTICIONES_A_EJECUTAR = [0]`. Cifras copiadas de la salida del notebook. Entrenamiento: 41,8 minutos.

| Medición | Exactitud | Top-5 | F1 macro | Exactitud en pares nuevos |
|---|---|---|---|---|
| Como la entrega el script crudo | 0,0014 | 0,0063 | 0,0006 | 0,0012 |
| **Mismo modelo, predicciones alineadas** | **0,8214** | **0,9360** | **0,7068** | **0,7023** |

## El error

El script usa `group_by_length=True` para entrenar más rápido. En la librería instalada (transformers 4.x) esa opción también
reordena la evaluación: `Trainer._get_eval_sampler` devuelve un `LengthGroupedSampler`. El script supone que
`trainer.predict` devuelve las predicciones en el orden original y las compara con la fila equivocada.

Se notó porque la validación interna daba 0,826 (ahí las etiquetas vienen en el mismo orden que las predicciones) y la prueba
daba azar. Se comprobó cargando el modelo ya entrenado, sin reentrenar, con `group_by_length=False`.

## Cómo leer 0,8214

Está inflado: el 46,1 % de los pares concepto + respuesta de la prueba ya estaba en el entrenamiento. En los pares nuevos
(53,9 % de la prueba) la exactitud es 0,7023. Es una sola partición, así que no hay desviación estándar.
