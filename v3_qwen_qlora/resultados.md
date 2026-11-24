# Resultados de v3: Qwen3-4B + QLoRA con datos limpios

Corrida del 23/11/2026 en Colab (Tesla T4). Cifras copiadas de la salida del notebook; los archivos quedaron en
`resultados_v3` del Drive. Una sola semilla.

Es el modelo de la versión cruda (v0) con tres cambios: evaluación bien alineada, la misma partición sin duplicados de v1 y v2
(14.424 entrenamiento / 3.092 validación / 3.092 prueba, 0 % de pares repetidos) y 3 épocas en vez de 2. El script se arma con
`armar_v3.py`, que lista cada cambio.

| Medida | Validación | Prueba |
|---|---|---|
| Exactitud | 0,7173 | **0,7238** |
| Top-3 | – | 0,8564 |
| Top-5 | – | 0,8891 |
| F1 macro | – | 0,6343 |
| F1 ponderado | – | 0,7070 |
| Kappa | – | 0,7233 |

Parámetros entrenables: 37.592.064 de 4.064.622.080 (0,92 %). Tiempo total de la partición: 50,0 minutos.

## Comparación con la misma prueba sin duplicados (3.092 ejemplos)

| Sistema | Exactitud | Top-5 | F1 macro |
|---|---|---|---|
| v1: TF-IDF + kNN | 0,5828 | 0,7015 | 0,4967 |
| v2: BETO + LoRA (3 semillas) | 0,6529 ± 0,0011 | 0,8282 ± 0,0033 | 0,5569 ± 0,0011 |
| **v3: Qwen3-4B + QLoRA (1 semilla)** | **0,7238** | **0,8891** | **0,6343** |

Con 3.092 ejemplos, el error estándar de cada exactitud es de unos 0,008 a 0,009.

## Limitaciones

- Una semilla y una partición: no hay desviación estándar.
- Las 3 épocas se fijaron antes de correr, sin buscar hiperparámetros; v2 sí comparó 7 variantes.
- El 2,1 % de la prueba tiene códigos que no están en el entrenamiento.
- v3 pesa y demora mucho más que v2 (50 minutos contra 4 por entrenamiento), a cambio de unos 7 puntos de exactitud.
