# Rol
Actúa como un ingeniero de Procesamiento del Lenguaje Natural con experiencia en ajuste fino eficiente (PEFT) de modelos de lenguaje abiertos.

# Contexto
En los estudios de memoria semántica se usa la tarea de listado de propiedades: a una persona se le da un concepto (por ejemplo, "caracol") y escribe propiedades de ese concepto (por ejemplo, "baboso"). Después, cada respuesta se clasifica con un código que agrupa respuestas de significado parecido. Hoy ese trabajo lo hacen personas y queremos automatizarlo.

Tenemos el archivo `CPN120.csv`, en español de Chile, separado por comas y en UTF-8, con 31.864 filas y tres columnas:
- `Concepto`: el concepto que se le mostró a la persona (120 distintos).
- `Respuesta`: la propiedad que escribió, en texto libre.
- `Codigo`: la etiqueta correcta que hay que predecir (1.785 distintas).

Ejemplos, entre las etiquetas <ejemplos>:

<ejemplos>

| Concepto | Respuesta | Codigo |
|---|---|---|
| granito | raspa | lastimar |
| caracol | autoprotección | proteccion |
| cáscara | protege a la comida | proteccion |
| ajedrez | se usan 16 piezas móviles | piezas_tablero |

</ejemplos>

El sistema de referencia que queremos superar (Ramos et al., 2025) ajustó GPT-3.5-turbo usando las respuestas y obtuvo una exactitud (accuracy) de 0,582 en prueba con validación cruzada de 5 particiones.

# Tarea
Escribe el código completo para entrenar, con una técnica de PEFT, un modelo que prediga `Codigo` a partir de `Concepto` y `Respuesta`, y para medir qué tan bien lo hace.

# Restricciones
- Debe correr en Google Colab gratuito, con una GPU T4.
- Solo modelos abiertos que se ejecuten localmente, sin APIs de pago. Se sugiere `qwen3:4b` o `gemma3:4b`, pero puedes elegir otro si lo justificas.
- Código en Python, con comentarios en español.
- Antes de entregar, verifica que el script esté completo y que las columnas que usa existan en el CSV.

# Formato de salida
Entrega un único script de Python dividido en celdas con `# %%`, en este orden:
1. Instalación e importaciones
2. Carga y preparación de los datos
3. Modelo y configuración PEFT
4. Entrenamiento
5. Evaluación con métricas
6. Ejemplo de predicción

Al final, explica en un párrafo corto cada decisión importante: modelo, técnica de PEFT, partición de los datos y métricas.
