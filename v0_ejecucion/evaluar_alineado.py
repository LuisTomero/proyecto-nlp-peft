# -*- coding: utf-8 -*-
"""
Codificación automática de respuestas de listado de propiedades (CPN120).

Predice `Codigo` a partir de `Concepto` y `Respuesta` ajustando Qwen3-4B-Base con
QLoRA (LoRA sobre un modelo cuantizado a 4 bits) y una cabeza de clasificación.

Sistema de referencia: Ramos et al. (2025), GPT-3.5-turbo ajustado,
accuracy = 0,582 en validación cruzada de 5 particiones.

Uso en Google Colab (gratuito):
  1. Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4.
  2. Deja CPN120.csv en "Mi unidad" de Google Drive (o súbelo cuando se pida).
  3. Ejecuta las celdas en orden. Si la sesión se corta, vuelve a ejecutarlas:
     las particiones ya terminadas quedan guardadas en Drive y se saltan.
"""

# %% [1] Instalación e importaciones
import subprocess
import sys


def instalar_dependencias():
    """Instala las librerías que Colab no trae o trae en versiones antiguas."""
    paquetes = [
        "transformers>=4.51.3,<5",  # 4.51.3 es la primera versión con Qwen3
        "peft>=0.15.0,<1",          # LoRA / QLoRA
        "bitsandbytes>=0.46.0",     # cuantización a 4 bits
        "accelerate>=1.6.0",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *paquetes], check=True)


instalar_dependencias()

import gc
import json
import os
import re
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

SEMILLA = 42
set_seed(SEMILLA)  # fija las semillas de random, numpy y torch

# ¿Estamos en Colab? (sirve para montar Drive o subir archivos)
try:
    import google.colab  # noqa: F401
    EN_COLAB = True
except ImportError:
    EN_COLAB = False

HAY_GPU = torch.cuda.is_available()
if HAY_GPU:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | memoria: {props.total_memory / 1e9:.1f} GB")
else:
    warnings.warn("No se detectó GPU. En Colab elige 'GPU T4'; en CPU el entrenamiento no es viable.")


# %% [2] Carga y preparación de los datos
USAR_DRIVE = False            # guarda resultados en Drive para poder retomar si se corta la sesión
RUTA_CSV = "CPN120.csv"
DIR_RESULTADOS = "resultados_cpn120_corregido"
N_PARTICIONES = 5            # igual que el sistema de referencia

if USAR_DRIVE and EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    RUTA_CSV = "/content/drive/MyDrive/CPN120.csv"
    DIR_RESULTADOS = "/content/drive/MyDrive/resultados_cpn120"

# Si el CSV no está donde se espera, se pide subirlo
if not os.path.exists(RUTA_CSV) and EN_COLAB:
    from google.colab import files
    print(f"No encontré {RUTA_CSV}. Sube CPN120.csv:")
    subidos = files.upload()
    RUTA_CSV = next(iter(subidos))
if not os.path.exists(RUTA_CSV):
    raise FileNotFoundError(f"No existe el archivo {RUTA_CSV}")
os.makedirs(DIR_RESULTADOS, exist_ok=True)


def sin_tildes(texto):
    """Quita tildes y diacríticos (á→a, ñ→n). Solo se usa para comparar, no para el modelo."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def limpiar(texto):
    """Normaliza Unicode y espacios sin cambiar el contenido del texto."""
    texto = unicodedata.normalize("NFC", str(texto))
    return re.sub(r"\s+", " ", texto).strip()


# Lectura: todo como texto y sin convertir "NA", "null", etc. en valores faltantes
# (una persona podría haber escrito literalmente "NA" o "nada").
df_crudo = pd.read_csv(RUTA_CSV, sep=",", encoding="utf-8-sig", dtype=str, keep_default_na=False)

# Verificación de columnas. Se toleran diferencias de mayúsculas, espacios o tildes
# ("Código" en vez de "Codigo"), pero las tres columnas deben existir.
COLUMNAS = {"concepto": "Concepto", "respuesta": "Respuesta", "codigo": "Codigo"}
renombrar = {}
for col in df_crudo.columns:
    clave = sin_tildes(str(col)).strip().lower()
    if clave in COLUMNAS:
        renombrar[col] = COLUMNAS[clave]
df_crudo = df_crudo.rename(columns=renombrar)
faltantes = [c for c in COLUMNAS.values() if c not in df_crudo.columns]
if faltantes:
    raise ValueError(f"Faltan las columnas {faltantes}. Columnas del CSV: {list(df_crudo.columns)}")

df = df_crudo[list(COLUMNAS.values())].copy()
for col in df.columns:
    df[col] = df[col].map(limpiar)

# Se descartan filas sin concepto, respuesta o código (no se pueden usar para entrenar ni evaluar)
vacias = (df["Concepto"] == "") | (df["Respuesta"] == "") | (df["Codigo"] == "")
if vacias.any():
    print(f"Se eliminan {int(vacias.sum())} filas con campos vacíos.")
df = df[~vacias].reset_index(drop=True)

print(f"Filas: {len(df):,} | Conceptos: {df['Concepto'].nunique()} | Códigos: {df['Codigo'].nunique():,}")
for nombre, obtenido, esperado in [
    ("filas", len(df), 31864),
    ("conceptos", df["Concepto"].nunique(), 120),
    ("códigos", df["Codigo"].nunique(), 1785),
]:
    if obtenido != esperado:
        print(f"  Aviso: se esperaban {esperado:,} {nombre} y hay {obtenido:,}.")

frecuencia_codigos = df["Codigo"].value_counts()
print(f"Códigos con 1 sola respuesta: {(frecuencia_codigos == 1).sum()} | "
      f"con menos de {N_PARTICIONES}: {(frecuencia_codigos < N_PARTICIONES).sum()}")


def construir_entrada(concepto, respuesta):
    """Texto que ve el modelo. Termina en 'Código:' para que el último token resuma el par."""
    return f"Concepto: {concepto}\nRespuesta: {respuesta}\nCódigo:"


def clave_par(concepto, respuesta):
    """Clave normalizada del par concepto–respuesta para detectar respuestas repetidas."""
    texto = sin_tildes(f"{concepto} || {respuesta}".lower())
    texto = re.sub(r"[^a-z0-9| ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


df["texto"] = [construir_entrada(c, r) for c, r in zip(df["Concepto"], df["Respuesta"])]
df["clave"] = [clave_par(c, r) for c, r in zip(df["Concepto"], df["Respuesta"])]

# Codificación de etiquetas: cada código es una clase
CODIGOS = sorted(df["Codigo"].unique())
codigo2id = {c: i for i, c in enumerate(CODIGOS)}
id2codigo = {i: c for c, i in codigo2id.items()}
N_CLASES = len(CODIGOS)
df["etiqueta"] = df["Codigo"].map(codigo2id).astype(int)

# Validación cruzada estratificada de 5 particiones. Los códigos con menos de 5
# respuestas no pueden repartirse en todas las particiones (sklearn avisa y sigue).
# Las particiones se guardan para que sean idénticas si hay que retomar.
ruta_particiones = os.path.join(DIR_RESULTADOS, "particiones.csv")
particiones_reutilizadas = False
if os.path.exists(ruta_particiones):
    guardadas = pd.read_csv(ruta_particiones, dtype=str, keep_default_na=False)
    if len(guardadas) == len(df) and (guardadas[["Concepto", "Respuesta", "Codigo"]].values
                                      == df[["Concepto", "Respuesta", "Codigo"]].values).all():
        df["particion"] = guardadas["particion"].astype(int).values
        particiones_reutilizadas = True
        print("Particiones reutilizadas desde", ruta_particiones)
    else:
        print("Aviso: particiones.csv no coincide con los datos; se recalculan. "
              "Borra las predicciones antiguas de DIR_RESULTADOS para no mezclar resultados.")
if not particiones_reutilizadas:
    df["particion"] = -1
    skf = StratifiedKFold(n_splits=N_PARTICIONES, shuffle=True, random_state=SEMILLA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, idx_prueba) in enumerate(skf.split(np.zeros(len(df)), df["etiqueta"])):
            df.loc[idx_prueba, "particion"] = k
    df[["Concepto", "Respuesta", "Codigo", "particion"]].to_csv(ruta_particiones, index=False)

# Diagnóstico por partición: tamaño, cobertura de códigos y pares repetidos
for k in range(N_PARTICIONES):
    ent, pru = df[df["particion"] != k], df[df["particion"] == k]
    cobertura = pru["Codigo"].isin(set(ent["Codigo"])).mean()
    repetidos = pru["clave"].isin(set(ent["clave"])).mean()
    print(f"Partición {k}: prueba={len(pru):,} | código visto en entrenamiento: {cobertura:.1%} | "
          f"par concepto–respuesta ya visto: {repetidos:.1%}")


# %% [3] Modelo y configuración PEFT
NOMBRE_MODELO = "Qwen/Qwen3-4B-Base"  # alternativa más rápida: "Qwen/Qwen3-1.7B-Base"
MAX_LONGITUD = 64                     # las entradas son cortas; se revisa abajo
GRADIENT_CHECKPOINTING = True         # ahorra memoria; en False entrena más rápido si cabe

tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Revisión de longitudes para elegir MAX_LONGITUD sin cortar textos
longitudes = np.array([len(ids) for ids in tokenizer(df["texto"].tolist())["input_ids"]])
print(f"Tokens por entrada: mediana={np.median(longitudes):.0f} | p99={np.percentile(longitudes, 99):.0f} | "
      f"máx={longitudes.max()} | truncadas con {MAX_LONGITUD}: {(longitudes > MAX_LONGITUD).sum()}")


class DatasetCPN(torch.utils.data.Dataset):
    """Dataset ya tokenizado; el relleno (padding) se hace por lote en el collator."""

    def __init__(self, textos, etiquetas):
        self.codificacion = tokenizer(list(textos), truncation=True, max_length=MAX_LONGITUD)
        self.etiquetas = [int(e) for e in etiquetas]

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, i):
        return {
            "input_ids": self.codificacion["input_ids"][i],
            "attention_mask": self.codificacion["attention_mask"][i],
            "labels": self.etiquetas[i],
        }


collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# Cuantización a 4 bits (NF4) del modelo base. La cabeza de clasificación ("score")
# se deja sin cuantizar porque se entrena desde cero.
config_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # la T4 no tiene bfloat16 nativo
    llm_int8_skip_modules=["score"],
)

# LoRA en todas las proyecciones lineales de atención y MLP.
# La cabeza "score" (1.785 salidas) se entrena completa (modules_to_save).
config_lora = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["score"],
)


def cargar_modelo_base():
    """Carga Qwen3 cuantizado con una cabeza de clasificación de N_CLASES salidas."""
    base = AutoModelForSequenceClassification.from_pretrained(
        NOMBRE_MODELO,
        num_labels=N_CLASES,
        id2label=id2codigo,
        label2id=codigo2id,
        quantization_config=config_bnb,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    base.config.pad_token_id = tokenizer.pad_token_id  # necesario para ubicar el último token real
    base.config.use_cache = False
    return base


def crear_modelo():
    """Modelo base cuantizado + adaptadores LoRA nuevos (QLoRA)."""
    base = cargar_modelo_base()
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return get_peft_model(base, config_lora)


def crear_optimizador(modelo, lr_lora, lr_cabeza):
    """AdamW con dos tasas: la cabeza parte de cero y aprende más rápido que los adaptadores."""
    params_lora, params_cabeza = [], []
    for nombre, param in modelo.named_parameters():
        if param.requires_grad:
            (params_cabeza if "score" in nombre else params_lora).append(param)
    return torch.optim.AdamW(
        [{"params": params_lora, "lr": lr_lora}, {"params": params_cabeza, "lr": lr_cabeza}],
        weight_decay=0.0,
    )


# Se crea un primer modelo para revisar la configuración; se reutiliza en la primera partición.
modelo = PeftModel.from_pretrained(cargar_modelo_base(), "resultados_cpn120/adaptador_particion0")
modelo.print_trainable_parameters()
modelo_sin_entrenar = True
if HAY_GPU:
    print(f"Memoria GPU en uso: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# %% [4] Entrenamiento (una corrida por partición; se puede retomar)
NUM_EPOCAS = 2
TAM_LOTE = 32
ACUM_GRADIENTES = 1
LR_LORA = 2e-4
LR_CABEZA = 1e-3
SUAVIZADO_ETIQUETAS = 0.1      # ayuda con 1.785 clases y códigos parecidos entre sí
FRACCION_VALIDACION = 0.05     # sale del entrenamiento; la partición de prueba no se toca
PARTICIONES_A_EJECUTAR = [0]
GUARDAR_ADAPTADORES = False     # ~150 MB por partición
DIR_TEMPORAL = "/tmp/cpn120_entrenamiento"


def ruta_predicciones(k):
    return os.path.join(DIR_RESULTADOS, f"predicciones_particion{k}.csv")


def ruta_adaptador(k):
    return os.path.join(DIR_RESULTADOS, f"adaptador_particion{k}")


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def metricas_validacion(eval_pred):
    logits, etiquetas = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    return {"accuracy": float((np.argmax(logits, axis=-1) == etiquetas).mean())}


pendientes = [k for k in PARTICIONES_A_EJECUTAR if not os.path.exists(ruta_predicciones(k))]
print("Particiones pendientes:", pendientes or "ninguna")
particion_modelo = None  # partición cuyo modelo queda en memoria para la celda 6

for k in pendientes:
    inicio = time.time()
    print(f"\n=== Partición {k + 1}/{N_PARTICIONES} ===")
    entrenamiento = df[df["particion"] != k]
    prueba = df[df["particion"] == k]
    ent, val = train_test_split(entrenamiento, test_size=FRACCION_VALIDACION, random_state=SEMILLA)
    ds_ent = DatasetCPN(ent["texto"], ent["etiqueta"])
    ds_val = DatasetCPN(val["texto"], val["etiqueta"])
    ds_pru = DatasetCPN(prueba["texto"], prueba["etiqueta"])

    # Cada partición parte de un modelo nuevo (sin nada aprendido de otra partición)
    if modelo is None or not modelo_sin_entrenar:
        modelo = crear_modelo()
    modelo_sin_entrenar = False

    argumentos = TrainingArguments(
        output_dir=os.path.join(DIR_TEMPORAL, f"particion{k}"),
        num_train_epochs=NUM_EPOCAS,
        per_device_train_batch_size=TAM_LOTE,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=ACUM_GRADIENTES,
        learning_rate=LR_LORA,  # referencia para el registro; las tasas reales van en el optimizador
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        label_smoothing_factor=SUAVIZADO_ETIQUETAS,
        fp16=HAY_GPU,
        group_by_length=False,   # agrupa textos de largo parecido: menos relleno, más velocidad
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
        seed=SEMILLA,
    )
    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=ds_ent,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=metricas_validacion,
        optimizers=(crear_optimizador(modelo, LR_LORA, LR_CABEZA), None),
    )
    pass

    # Predicción sobre la partición de prueba (nunca vista durante el entrenamiento)
    salida = trainer.predict(ds_pru)
    logits = salida.predictions[0] if isinstance(salida.predictions, tuple) else salida.predictions
    probs = softmax(logits.astype(np.float32))
    top5 = np.argsort(-probs, axis=1)[:, :5]

    if GUARDAR_ADAPTADORES:
        trainer.model.save_pretrained(ruta_adaptador(k))
    with open(os.path.join(DIR_RESULTADOS, f"historial_particion{k}.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=1)

    predicciones = pd.DataFrame({
        "indice": prueba.index,
        "Concepto": prueba["Concepto"].values,
        "Respuesta": prueba["Respuesta"].values,
        "Codigo": prueba["Codigo"].values,
        "Prediccion": [id2codigo[i] for i in top5[:, 0]],
        "Confianza": probs[np.arange(len(probs)), top5[:, 0]],
        "Top5": [json.dumps([id2codigo[i] for i in fila], ensure_ascii=False) for fila in top5],
        "particion": k,
    })
    # Se escribe al final: si el archivo existe, la partición está completa
    predicciones.to_csv(ruta_predicciones(k), index=False)

    acc = (predicciones["Prediccion"] == predicciones["Codigo"]).mean()
    print(f"Partición {k}: accuracy en prueba = {acc:.4f} | tiempo = {(time.time() - inicio) / 60:.1f} min")
    particion_modelo = k

    # Se libera memoria salvo en la última partición (su modelo se usa en la celda 6)
    del trainer
    if k != pendientes[-1]:
        del modelo
        modelo = None
        gc.collect()
        if HAY_GPU:
            torch.cuda.empty_cache()


# %% [5] Evaluación con métricas
ACCURACY_REFERENCIA = 0.582  # Ramos et al. (2025), GPT-3.5-turbo ajustado


def accuracy_top_k(verdaderos, listas_top, k):
    return float(np.mean([v in lista[:k] for v, lista in zip(verdaderos, listas_top)]))


def prediccion_por_memoria(entrenamiento, prueba):
    """Línea base sin aprendizaje: el código más frecuente para el mismo par concepto–respuesta
    en entrenamiento; si el par es nuevo, el código más frecuente de ese concepto."""
    conteo_par = (entrenamiento.groupby(["clave", "Codigo"]).size().reset_index(name="n")
                  .sort_values(["clave", "n"], ascending=[True, False]).drop_duplicates("clave"))
    por_par = dict(zip(conteo_par["clave"], conteo_par["Codigo"]))
    conteo_con = (entrenamiento.groupby(["Concepto", "Codigo"]).size().reset_index(name="n")
                  .sort_values(["Concepto", "n"], ascending=[True, False]).drop_duplicates("Concepto"))
    por_concepto = dict(zip(conteo_con["Concepto"], conteo_con["Codigo"]))
    mas_frecuente = entrenamiento["Codigo"].value_counts().index[0]
    return np.array([por_par.get(cl, por_concepto.get(co, mas_frecuente))
                     for cl, co in zip(prueba["clave"], prueba["Concepto"])])


filas_metricas, todas = [], []
for k in range(N_PARTICIONES):
    if not os.path.exists(ruta_predicciones(k)):
        continue
    pred = pd.read_csv(ruta_predicciones(k), keep_default_na=False,
                       dtype={"Codigo": str, "Prediccion": str})
    pred["Top5"] = pred["Top5"].map(json.loads)
    prueba = df.loc[pred["indice"].values]
    entrenamiento = df[df["particion"] != k]
    assert (prueba["Codigo"].values == pred["Codigo"].values).all(), "Predicciones desalineadas con los datos"

    y, y_hat = pred["Codigo"].values, pred["Prediccion"].values
    nuevos = ~prueba["clave"].isin(set(entrenamiento["clave"])).values
    filas_metricas.append({
        "particion": k,
        "n_prueba": len(pred),
        "accuracy": accuracy_score(y, y_hat),
        "top3": accuracy_top_k(y, pred["Top5"], 3),
        "top5": accuracy_top_k(y, pred["Top5"], 5),
        "f1_macro": f1_score(y, y_hat, labels=np.unique(y), average="macro", zero_division=0),
        "f1_ponderado": f1_score(y, y_hat, labels=np.unique(y), average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y, y_hat),
        "cobertura_codigos": float(np.isin(y, entrenamiento["Codigo"].unique()).mean()),
        "accuracy_pares_nuevos": float((y[nuevos] == y_hat[nuevos]).mean()) if nuevos.any() else np.nan,
        "prop_pares_nuevos": float(nuevos.mean()),
        "accuracy_linea_base_memoria": accuracy_score(y, prediccion_por_memoria(entrenamiento, prueba)),
    })
    todas.append(pred)

if not filas_metricas:
    raise RuntimeError("No hay predicciones guardadas. Ejecuta primero la celda 4.")

metricas = pd.DataFrame(filas_metricas).set_index("particion")
pd.set_option("display.width", 200)
print("Métricas por partición:\n", metricas.round(4).to_string())

resumen = metricas.drop(columns="n_prueba").agg(["mean", "std"]).T
resumen.columns = ["media", "desv_est"]
print("\nResumen (media ± desviación estándar entre particiones):\n", resumen.round(4).to_string())
resumen.to_csv(os.path.join(DIR_RESULTADOS, "resumen_metricas.csv"))
metricas.to_csv(os.path.join(DIR_RESULTADOS, "metricas_por_particion.csv"))

# Comparación con la referencia (intervalo t de 95 % sobre las particiones)
n = len(metricas)
media_acc = metricas["accuracy"].mean()
print(f"\nAccuracy: {media_acc:.4f} vs. referencia {ACCURACY_REFERENCIA:.3f} "
      f"(diferencia {media_acc - ACCURACY_REFERENCIA:+.4f})")
if n >= 2:
    ic = stats.t.interval(0.95, n - 1, loc=media_acc, scale=metricas["accuracy"].std(ddof=1) / np.sqrt(n))
    print(f"IC 95 % de la accuracy: [{ic[0]:.4f}, {ic[1]:.4f}]")
if n < N_PARTICIONES:
    print(f"Aviso: solo hay {n} de {N_PARTICIONES} particiones; la comparación es preliminar.")

# Análisis de errores con todas las particiones juntas
total = pd.concat(todas, ignore_index=True)
total["acierto"] = total["Codigo"] == total["Prediccion"]

por_concepto = total.groupby("Concepto")["acierto"].agg(["mean", "size"]).sort_values("mean")
print("\nConceptos con menor accuracy:\n", por_concepto.head(10).round(3).to_string())

errores = total[~total["acierto"]].groupby(["Codigo", "Prediccion"]).size().sort_values(ascending=False)
print("\nConfusiones más frecuentes (código real → predicho):\n", errores.head(15).to_string())

# Accuracy según la confianza del modelo: útil para aceptar automáticamente
# las predicciones seguras y enviar el resto a revisión humana.
print("\nUmbral de confianza | % de respuestas cubiertas | accuracy en esas respuestas")
for umbral in [0.0, 0.5, 0.7, 0.9]:
    sel = total["Confianza"] >= umbral
    if sel.any():
        print(f"  ≥ {umbral:.1f}              | {sel.mean():6.1%}                  | {total.loc[sel, 'acierto'].mean():.4f}")


# %% [6] Ejemplo de predicción
def cargar_modelo_guardado(k):
    """Reconstruye el modelo de la partición k desde su adaptador guardado."""
    base = cargar_modelo_base()
    return PeftModel.from_pretrained(base, ruta_adaptador(k))


# Usa el modelo que quedó en memoria; si la sesión se reinició, carga el último guardado
if modelo is None or particion_modelo is None:
    guardados = [k for k in range(N_PARTICIONES) if os.path.isdir(ruta_adaptador(k))]
    if not guardados:
        raise RuntimeError("No hay modelos guardados. Ejecuta la celda 4 con GUARDAR_ADAPTADORES = True.")
    particion_modelo = guardados[-1]
    modelo = cargar_modelo_guardado(particion_modelo)
modelo.eval()


@torch.no_grad()
def predecir_codigos(concepto, respuesta, k=3):
    """Devuelve los k códigos más probables con su probabilidad."""
    texto = construir_entrada(limpiar(concepto), limpiar(respuesta))
    entrada = tokenizer(texto, return_tensors="pt", truncation=True, max_length=MAX_LONGITUD)
    entrada = entrada.to(next(modelo.parameters()).device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=HAY_GPU):
        logits = modelo(**entrada).logits.float()
    probs = torch.softmax(logits, dim=-1)[0]
    mejores = torch.topk(probs, k)
    return [(id2codigo[i], p) for i, p in zip(mejores.indices.tolist(), mejores.values.tolist())]


# Los cuatro ejemplos del enunciado y dos respuestas inventadas.
# Se indica si el par estaba en el entrenamiento de este modelo (si estaba, no es una evaluación).
claves_entrenamiento = set(df.loc[df["particion"] != particion_modelo, "clave"])
ejemplos = [
    ("granito", "raspa", "lastimar"),
    ("caracol", "autoprotección", "proteccion"),
    ("cáscara", "protege a la comida", "proteccion"),
    ("ajedrez", "se usan 16 piezas móviles", "piezas_tablero"),
    ("caracol", "deja un rastro brillante", None),
    ("ajedrez", "lo juegan dos personas", None),
]
print(f"Modelo de la partición {particion_modelo}\n")
for concepto, respuesta, esperado in ejemplos:
    candidatos = predecir_codigos(concepto, respuesta)
    visto = "sí" if clave_par(concepto, respuesta) in claves_entrenamiento else "no"
    print(f"{concepto} | {respuesta}  (visto en entrenamiento: {visto})")
    if esperado:
        print(f"  esperado: {esperado}")
    for codigo, p in candidatos:
        print(f"  → {codigo:<25} {p:.3f}")
