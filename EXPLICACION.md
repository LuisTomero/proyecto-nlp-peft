# Explicación del código

Este documento recorre cada carpeta en el orden en que se hizo el trabajo y dice qué hace cada archivo y qué se
cambió respecto de la versión anterior. Los números salen de los `resultados.md` de cada carpeta.

## datos/

`CPN120.csv` es el archivo del curso: 31.864 filas con `Concepto`, `Respuesta` y `Codigo`.

`preparar_datos.py` hace dos cosas.

1. Quita los duplicados. Normaliza el concepto y la respuesta (minúsculas, sin tildes pero conservando la ñ, sin
   signos ni espacios repetidos) y deja una fila por cada par normalizado. Si un mismo par tiene varios códigos, se
   queda con el más frecuente; en caso de empate, con el primero en orden alfabético. Quedan 20.608 filas.
2. Parte esas filas en entrenamiento (14.424), validación (3.092) y prueba (3.092), con semilla 42. Un `assert`
   comprueba que ningún par aparezca en dos partes.

Esto se hizo porque en el CSV original muchas personas escribieron lo mismo. Con una partición al azar sobre todas las
filas, casi la mitad de la prueba ya está en el entrenamiento (46,1 % en la partición 0 de v0) y la exactitud sale
inflada. El resultado queda en `cpn120_particion.csv`. v2 y v3 recalculan la partición y comprueban con un hash
SHA-256 que sea idéntica a la de este archivo, así las tres versiones se miden con los mismos 3.092 ejemplos.

Aun así, el 2,1 % de la prueba tiene un código que no aparece en el entrenamiento, y ahí ningún modelo puede acertar.

## v0_cruda/

Es lo que entregó Claude con un solo prompt, sin cambios: `prompt.md` (lo que se pidió), `respuesta_cruda.md` (lo que
respondió) y `cpn120_qlora.py` (el script, de 580 líneas, con su hash en el README).

El script tiene las seis celdas que pedía el prompt:

1. Instalación e importaciones.
2. Carga y limpieza de datos. Arma el texto de entrada como `Concepto: ...\nRespuesta: ...\nCódigo:` y crea una clase
   por cada código (1.785).
3. Modelo y PEFT. Qwen3-4B-Base cargado a 4 bits (NF4, con doble cuantización y cálculo en fp16 porque la T4 no tiene
   bfloat16). LoRA con r=16, alpha=32 y dropout 0,05 en las siete proyecciones lineales (atención y MLP). Encima del
   modelo va una cabeza de clasificación con 1.785 salidas que se entrena completa. Usa suavizado de etiquetas de 0,1.
4. Entrenamiento con validación cruzada de 5 particiones, igual que el paper, 2 épocas por partición.
5. Evaluación: exactitud, top-3, top-5, F1 macro y ponderado, kappa, y una línea base que solo memoriza el código más
   frecuente de cada par.
6. Ejemplo de predicción.

## v0_ejecucion/

Aquí se corrió v0 y se encontró un error.

- `v0_cruda_colab.ipynb` y `v0_cruda_kaggle.ipynb` son el mismo script separado en celdas. Los únicos cambios son
  `USAR_DRIVE = False` y `PARTICIONES_A_EJECUTAR = [0]` (solo una de las cinco particiones, por el límite de tiempo
  de Colab). El de Kaggle agrega una celda inicial. `armar_kaggle.py` genera el segundo notebook.
- El script crudo dio 0,0014 de exactitud. La causa es `group_by_length=True`: en la versión instalada de
  transformers esa opción reordena también la evaluación, y el script compara cada predicción con la fila equivocada.
  La validación interna daba 0,826 y la prueba daba azar, y eso fue lo que delató el problema.
- `armar_evaluacion_alineada.py` genera `evaluar_alineado.py`, que carga el modelo ya entrenado, sin reentrenar, con
  `group_by_length=False`. El resultado es 0,8214 de exactitud, o 0,7023 si solo se cuentan los pares que el modelo no
  había visto. El número está inflado porque el 46,1 % de la prueba estaba en el entrenamiento.

Los scripts `armar_*.py` reemplazan texto del original y verifican con un `assert` cuántas veces aparece cada trozo
reemplazado. Así el script crudo no se toca y cada modificación queda escrita en una lista.

## v1_tfidf/

`tfidf_knn.py` reemplaza el modelo grande por una técnica simple, como pide el enunciado.

- Cada respuesta se convierte en un vector TF-IDF de trozos de 2 a 5 letras. El concepto tiene su propio TF-IDF, que se
  suma con un peso.
- Para clasificar, se buscan los k vecinos más parecidos del entrenamiento y cada uno vota por su código con peso
  igual a su similitud.
- k (1, 3, 5, 10) y el peso del concepto (de 0 a 1,5) se eligen solo con validación. Quedó k=3 y peso 0,6.
- En la prueba sin duplicados da 0,583 (0,534 si solo se usa la respuesta). En el esquema del paper, con 5 particiones
  y pares repetidos, da 0,740. La diferencia entre ambos números muestra cuánto inflan los duplicados.

Corre en cualquier computador, sin GPU.

## v2_beto_lora/

`beto_lora.py` usa BETO (BERT en español) con LoRA, partiendo de lo visto en la Clase 11. La entrada es el par
(concepto, respuesta), con un máximo de 48 tokens, y encima hay una capa de clasificación.

La configuración base (r=8, alpha=16, dropout 0,1, LoRA en query y value, 12 épocas, lote de 64) viene de la clase.
Después se probaron seis variantes (E1 a E6) en validación: entrenar el pooler, quitar el concepto, r=16, r=32, LoRA
en más capas (query, key, value y dense) y una tasa de aprendizaje menor. La mejor fue E5, y con esa se hizo la
prueba final con tres semillas: 0,653 ± 0,001.

`armar_notebook.py` convierte el script en `beto_lora_colab.ipynb`. Las diferencias entre E1, E3, E4, E5 y E6 son de una
sola semilla y están dentro del margen de error (unos 0,009), así que no se puede afirmar cuál es mejor entre ellas. Lo
que sí se ve es que quitar el concepto empeora (E2).

## v3_qwen_qlora/

`armar_v3.py` genera `v3_qwen_qlora.py` a partir del script crudo. Los cambios están en la lista `CAMBIOS`, con su
motivo:

- Los datos usan la partición sin duplicados de v1 y v2 (con el hash verificado), en vez de la validación cruzada.
- `group_by_length=False`, que corrige el error de v0.
- 3 épocas en vez de 2. Con menos filas de entrenamiento, 3 épocas dan 1.353 pasos, parecido a los 1.514 de v0.
- Ajustes de ejecución: sin Drive y con carpeta de resultados propia.

Es el mismo modelo de v0 (Qwen3-4B-Base con QLoRA y cabeza de clasificación). Da 0,724 de exactitud, 0,889 de top-5 y
0,634 de F1 macro, con un solo entrenamiento de 50 minutos en la T4. No se probaron otros hiperparámetros ni otras
semillas.
