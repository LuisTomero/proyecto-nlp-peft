Voy a escribir el script, revisar su sintaxis y probar todo el flujo con un CSV sintético que tiene las mismas columnas. Así compruebo que corre de punta a punta antes de entregártelo.
Te envié el script `cpn120_qlora.py`: entrena Qwen3-4B-Base con QLoRA para predecir `Codigo` y lo evalúa con las mismas 5 particiones que Ramos et al. Todavía no sé si supera el 0,582, porque no tengo `CPN120.csv` ni una GPU. Lo corrí de punta a punta en CPU con un CSV inventado (mismas columnas, incluido un código "NA" y una respuesta vacía) y un Qwen3 diminuto sin entrenar. Las 6 celdas funcionaron, también retomar un trabajo a medias y cargar un adaptador guardado. Esa prueba solo verifica que el código corre; la exactitud real aparece recién al usarlo con tus datos en una T4.
Sobre las columnas: el script revisa que existan `Concepto`, `Respuesta` y `Codigo`, y se detiene con un error claro si falta alguna. Acepta diferencias de mayúsculas o tildes, como "Código". También avisa si el archivo no tiene 31.864 filas, 120 conceptos y 1.785 códigos.
No medí cuánto demora cada partición. Las 5 probablemente no alcanzan en una sola sesión gratuita de Colab. Por eso cada partición terminada se guarda en Drive y, si se corta la sesión, al volver a ejecutar se salta lo ya hecho. Si va lento, cambia `NOMBRE_MODELO` a `Qwen/Qwen3-1.7B-Base`.
Modelo. Elegí Qwen3-4B-Base, la versión de Hugging Face de tu sugerencia `qwen3:4b`. Las etiquetas de Ollama sirven para usar un modelo, no para entrenarlo. Descarté gemma3:4b por tres razones: incluye un componente de visión que ocupa memoria sin aportar nada aquí, su descarga requiere aceptar la licencia en Hugging Face, y se conocen problemas de desbordamiento numérico en fp16, que es la precisión que usa la T4. Qwen3 tiene licencia Apache 2.0, maneja bien el español y funciona en fp16. Usé la versión Base porque el modelo no conversa, solo clasifica.
Técnica de PEFT. Usé QLoRA: el modelo se carga comprimido a 4 bits y se entrenan solo unos adaptadores pequeños (rango 16) en todas las capas lineales. Así cabe en los 15 GB de la T4. En vez de que el modelo escriba el código como texto, le agregué una capa de clasificación con 1.785 salidas, que se entrena completa y con una tasa de aprendizaje más alta porque parte de cero. Esto trae tres ventajas:

* Siempre entrega un código que existe.
* Predice en una sola pasada, más rápido que generar texto.
* Da una probabilidad para cada código, lo que permite mostrar los 3 o 5 más probables y un nivel de confianza.

Además usé suavizado de etiquetas (0,1), porque hay muchos códigos parecidos entre sí.
Partición de los datos. Validación cruzada estratificada de 5 particiones con semilla fija, para comparar directamente con el 0,582. Las particiones se guardan en un archivo, así que al retomar se usan exactamente las mismas. Dentro de cada entrenamiento se aparta un 5 % para seguir el avance, y la partición de prueba no se toca. Hay dos cosas a tener en cuenta:

* Los códigos con menos de 5 respuestas no alcanzan a quedar en todas las particiones. El script muestra cuántas respuestas de prueba tienen un código que el modelo nunca vio al entrenar.
* Muchas personas escriben lo mismo ("tiene patas"), y una partición al azar pone respuestas repetidas en entrenamiento y en prueba. Eso infla la exactitud, así que el script también la calcula solo sobre pares concepto–respuesta que el modelo no vio.

Métricas.

* Accuracy (exactitud): la métrica principal y la única comparable con el 0,582. Se informa como media ± desviación estándar entre particiones, con un intervalo de confianza de 95 %.
* Top-3 y top-5: si el código correcto está entre los primeros sugeridos, que es lo que importa si una persona elige entre sugerencias.
* F1 macro: pesa igual a los códigos poco frecuentes. El F1 ponderado y el kappa de Cohen se reportan como complemento; el kappa es la medida habitual de acuerdo entre codificadores.
* Comparación de referencia: una regla simple que copia el código más frecuente para la misma respuesta ya vista. Muestra cuánto aporta el modelo más allá de memorizar.
* Exactitud según confianza: indica qué porcentaje de respuestas se podría codificar automáticamente con buena exactitud y cuál conviene revisar a mano.
