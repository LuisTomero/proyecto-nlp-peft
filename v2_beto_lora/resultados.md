# Resultados de v2: BETO + LoRA

Corrida del 23/11/2026 en Colab (Tesla T4, torch 2.14.1, transformers 5.22.1, peft 0.21.2), 12 épocas.
Cifras copiadas de la salida del notebook. Los JSON completos quedaron en `resultados_v2` del Drive.

## Experimentos en validación (3.092 ejemplos sin duplicados, una semilla)

| Experimento | Exactitud | Top-5 | F1 macro | % entrenable | Minutos |
|---|---|---|---|---|---|
| E5: LoRA en query, key, value y dense | 0,6481 | 0,8305 | 0,5475 | 2,80 | 4,1 |
| E4: rango 32 | 0,6462 | 0,8250 | 0,5444 | 2,67 | 2,4 |
| E3: rango 16 | 0,6400 | 0,8263 | 0,5404 | 2,17 | 2,4 |
| E6: tasa de LoRA 3e-4 | 0,6394 | 0,8266 | 0,5373 | 1,91 | 2,4 |
| E1: entrena el pooler | 0,6365 | 0,8305 | 0,5361 | 1,91 | 2,4 |
| E0: configuración de la Clase 11 | 0,6255 | 0,8289 | 0,5161 | 1,40 | 2,9 |
| E2: E1 sin el concepto | 0,6129 | 0,7940 | 0,5115 | 1,91 | 2,3 |

Con 3.092 ejemplos, el error estándar de cada exactitud es de unos 0,009. Las diferencias entre E1, E3, E4, E5 y E6
(y la de E0 con E1) son de una sola semilla y están cerca de ese margen: no prueban cuál es mejor.

## Prueba final (3.092 ejemplos sin duplicados)

Configuración E5, elegida solo con validación. Media ± desviación estándar de 3 semillas.

| Sistema | Exactitud | Top-5 | F1 macro |
|---|---|---|---|
| BETO + LoRA (E5) | 0,6529 ± 0,0011 | 0,8282 ± 0,0033 | 0,5569 ± 0,0011 |
| TF-IDF + kNN (v1) | 0,5828 | 0,7015 | 0,4967 |

## 5 particiones con todas las filas (pares repetidos entre partes)

| Sistema | Exactitud |
|---|---|
| BETO + LoRA (E5) | 0,7773 ± 0,0036 |
| TF-IDF + kNN (v1) | 0,7399 ± 0,0049 |

Esta cifra está inflada por los duplicados. No es comparable directamente con el 0,582 de Ramos et al. (2025):
ellos informan 27.138 propiedades, el CSV del curso trae 31.864 filas y sus prompts muestran solo la respuesta, sin el concepto.

## Limitaciones

- Hay sobreajuste: la pérdida de entrenamiento termina cerca de 0,03.
- Se usaron 12 épocas fijas; en E5 la validación todavía subía levemente al final.
- El 2,1 % de la prueba tiene códigos que no están en el entrenamiento, así que ahí no se puede acertar.
