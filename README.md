# Proyecto NLP — Clasificación de propiedades con PEFT (CPN120)

Grupo 8 · Procesamiento del Lenguaje Natural · Universidad Adolfo Ibáñez

Dado un concepto y una propiedad escrita por una persona, predecir su código entre 1.785 posibles.
Sistema de referencia: Ramos et al. (2025), GPT-3.5-turbo ajustado, 0,582 de exactitud en CPN120.

## Contenido

| Carpeta | Qué contiene |
|---|---|
| `datos/` | `CPN120.csv`, entregado por el curso |
| `v0_cruda/` | Primera versión de la IA: el prompt, la respuesta y el script, sin modificaciones |
| `v0_ejecucion/` | Notebook para correr la versión cruda en Colab |

## Versión cruda (v0)

Se obtuvo con un solo prompt en claude.ai, en un chat nuevo y sin adjuntar archivos. `cpn120_qlora.py` es
el archivo descargado, sin cambios (SHA-256 `215c5af7130dbeba90d0dba564c7454af423d3d51af96e300a703c8a3e0a0206`).
Según su respuesta, Claude lo probó por su cuenta con datos sintéticos antes de entregarlo.

El notebook de `v0_ejecucion/` es el mismo script separado en celdas, con un solo cambio:
`PARTICIONES_A_EJECUTAR = [0]` en vez de las cinco, por el límite de tiempo de Colab gratuito.

## Uso de IA

La versión cruda se generó con Claude. También se usó Claude Code como apoyo en el resto del proyecto.

## Referencia

Ramos, D., Moreno, S., Canessa, E., y Chaigneau, S. (2025). Using Chat-GPT for coding properties in semantic
memory studies. ICPRS 2025.
