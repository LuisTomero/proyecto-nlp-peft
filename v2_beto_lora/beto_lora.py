# %% [1] Instalación e importaciones
# v2: BETO (BERT en español) + LoRA, a partir de la sección 7 del notebook de la Clase 11.
# Uso: Colab con GPU, y cpn120_particion.csv y CPN120.csv en "Mi unidad" de Google Drive.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("peft") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=True)

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
import peft
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELO = "dccuchile/bert-base-spanish-wwm-cased"
EQUIPO = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "|", EQUIPO)

# %% [2] Datos
CARPETA_DATOS, CARPETA_RES = ".", "resultados_v2"
if importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None:
    from google.colab import drive
    drive.mount("/content/drive")
    CARPETA_DATOS = "/content/drive/MyDrive"
    CARPETA_RES = "/content/drive/MyDrive/resultados_v2"   # si se corta la sesión, no se pierde lo hecho
os.makedirs(CARPETA_RES, exist_ok=True)

datos = pd.read_csv(f"{CARPETA_DATOS}/cpn120_particion.csv", dtype=str, keep_default_na=False)
ent = datos[datos["parte"] == "entrenamiento"].reset_index(drop=True)
val = datos[datos["parte"] == "validacion"].reset_index(drop=True)
pru = datos[datos["parte"] == "prueba"].reset_index(drop=True)
print("entrenamiento", len(ent), "| validación", len(val), "| prueba", len(pru))

tok = AutoTokenizer.from_pretrained(MODELO)


def tokenizar(df, usar_concepto):
    """Con concepto: [CLS] concepto [SEP] respuesta [SEP]. Sin concepto: [CLS] respuesta [SEP]."""
    if usar_concepto:
        enc = tok(df["Concepto"].tolist(), df["Respuesta"].tolist(), truncation=True, max_length=48)
    else:
        enc = tok(df["Respuesta"].tolist(), truncation=True, max_length=48)
    claves = [k for k in ("input_ids", "token_type_ids", "attention_mask") if k in enc]
    return [{k: enc[k][i] for k in claves} for i in range(len(df))]


def armar_lote(ejemplos):
    lote = tok.pad(ejemplos, padding=True, return_tensors="pt")
    return {k: v.to(EQUIPO) for k, v in lote.items()}


# %% [3] Modelo: BETO + LoRA
BASE = {
    "r": 8, "lora_alpha": 16, "lora_dropout": 0.1,   # valores del notebook de la Clase 11
    "target_modules": ["query", "value"],
    "entrenar_pooler": False,                        # como en el notebook: el pooler queda al azar y congelado
    "usar_concepto": True,
    "lr_lora": 1e-3, "lr_cabeza": 1e-3, "epocas": 12, "lote": 64,
}


def crear_base(n_clases):
    return AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=n_clases)


def crear_modelo(cfg, n_clases):
    """Congela BETO, agrega LoRA y deja entrenables la cabeza y, si se pide, el pooler."""
    guardar = ["classifier"] + (["pooler"] if cfg["entrenar_pooler"] else [])
    lora = LoraConfig(task_type="SEQ_CLS", r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], modules_to_save=guardar,
                      exclude_modules=r".*pooler.*" if cfg["entrenar_pooler"] else None)
    return get_peft_model(crear_base(n_clases), lora).to(EQUIPO)


def contar_parametros(modelo):
    return (sum(p.numel() for p in modelo.parameters() if p.requires_grad),
            sum(p.numel() for p in modelo.parameters()))


# %% [4] Entrenamiento y evaluación
def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predecir(modelo, ejemplos, codigos):
    """Los 5 códigos más probables de cada ejemplo."""
    modelo.eval()
    top = []
    for i in range(0, len(ejemplos), 256):
        with torch.autocast(EQUIPO, dtype=torch.float16, enabled=EQUIPO == "cuda"):
            logits = modelo(**armar_lote(ejemplos[i:i + 256])).logits.float()
        top += logits.topk(min(5, len(codigos)), dim=1).indices.tolist()
    return [[codigos[j] for j in fila] for fila in top]


def medir(reales, top5):
    top1 = [t[0] for t in top5]
    return {"accuracy": float(np.mean([r == p for r, p in zip(reales, top1)])),
            "top5": float(np.mean([r in t for r, t in zip(reales, top5)])),
            "f1_macro": float(f1_score(reales, top1, labels=np.unique(reales), average="macro", zero_division=0))}


def entrenar(cfg, df_ent, df_ev, semilla=0, medir_cada_epoca=True):
    """Entrena en df_ent y mide en df_ev. Las clases salen solo del entrenamiento."""
    fijar_semilla(semilla)
    inicio = time.time()
    codigos = sorted(df_ent["Codigo"].unique())
    a_id = {c: i for i, c in enumerate(codigos)}
    y = torch.tensor([a_id[c] for c in df_ent["Codigo"]])
    x_ent, x_ev = tokenizar(df_ent, cfg["usar_concepto"]), tokenizar(df_ev, cfg["usar_concepto"])
    reales = df_ev["Codigo"].tolist()

    modelo = crear_modelo(cfg, len(codigos))
    entrenables, total = contar_parametros(modelo)
    grupo_lora = [p for n, p in modelo.named_parameters() if p.requires_grad and "lora_" in n]
    grupo_cabeza = [p for n, p in modelo.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": grupo_lora, "lr": cfg["lr_lora"]},
                             {"params": grupo_cabeza, "lr": cfg["lr_cabeza"]}], weight_decay=0.01)
    pasos = cfg["epocas"] * ((len(x_ent) + cfg["lote"] - 1) // cfg["lote"])
    calent = max(1, int(0.05 * pasos))
    # Sube la tasa al comienzo (calentamiento) y después la baja en línea recta hasta 0
    plan = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda p: (p + 1) / calent if p < calent else max(0.0, (pasos - p) / max(1, pasos - calent)))
    escala = torch.amp.GradScaler(EQUIPO, enabled=EQUIPO == "cuda")

    curva = []
    for ep in range(cfg["epocas"]):
        modelo.train()
        perm, suma = torch.randperm(len(x_ent)).tolist(), 0.0
        for i in range(0, len(perm), cfg["lote"]):
            idx = perm[i:i + cfg["lote"]]
            with torch.autocast(EQUIPO, dtype=torch.float16, enabled=EQUIPO == "cuda"):
                logits = modelo(**armar_lote([x_ent[j] for j in idx])).logits
            perdida = F.cross_entropy(logits.float(), y[idx].to(EQUIPO))
            opt.zero_grad()
            escala.scale(perdida).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(grupo_lora + grupo_cabeza, 1.0)
            escala.step(opt)
            escala.update()
            plan.step()
            suma += perdida.item() * len(idx)
        linea = {"epoca": ep + 1, "perdida": suma / len(perm)}
        if medir_cada_epoca or ep == cfg["epocas"] - 1:
            top5 = predecir(modelo, x_ev, codigos)
            linea.update(medir(reales, top5))
        curva.append(linea)
        print("  ", {k: round(v, 4) for k, v in linea.items()}, flush=True)
    res = {"config": cfg, "semilla": semilla, "entrenables": entrenables, "total": total,
           "pct_entrenable": 100 * entrenables / total, "minutos": (time.time() - inicio) / 60,
           "final": {k: curva[-1][k] for k in ("accuracy", "top5", "f1_macro")}, "curva": curva}
    return modelo, codigos, top5, res


def correr(nombre, cfg, df_ent, df_ev, **kw):
    """Corre un experimento y guarda su resultado en Drive. Si ya estaba hecho, lo lee."""
    ruta = f"{CARPETA_RES}/{nombre}.json"
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            res = json.load(f)
        if res["config"] == cfg:
            print(nombre, "ya estaba hecho:", res["final"])
            return res
    print(nombre)
    modelo, _, _, res = entrenar(cfg, df_ent, df_ev, **kw)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    del modelo
    if EQUIPO == "cuda":
        torch.cuda.empty_cache()
    return res


# %% [5] Experimentos: se compara cada variante en VALIDACIÓN
# E1 cambia solo el pooler respecto de E0; E2 a E6 cambian una cosa respecto de E1.
EXPERIMENTOS = {
    "E0_clase11": {},
    "E1_pooler": {"entrenar_pooler": True},
    "E2_sin_concepto": {"entrenar_pooler": True, "usar_concepto": False},
    "E3_r16": {"entrenar_pooler": True, "r": 16, "lora_alpha": 32},
    "E4_r32": {"entrenar_pooler": True, "r": 32, "lora_alpha": 64},
    "E5_mas_capas": {"entrenar_pooler": True, "target_modules": ["query", "key", "value", "dense"]},
    "E6_lr_menor": {"entrenar_pooler": True, "lr_lora": 3e-4},
}
resultados = {n: correr(n, {**BASE, **c}, ent, val) for n, c in EXPERIMENTOS.items()}
tabla = pd.DataFrame([{"experimento": n, **r["final"], "% entrenable": r["pct_entrenable"], "min": r["minutos"]}
                      for n, r in resultados.items()]).sort_values("accuracy", ascending=False)
print(tabla.round(4).to_string(index=False))
tabla.to_csv(f"{CARPETA_RES}/tabla_validacion.csv", index=False)

# %% [6] Prueba final: la mejor configuración de validación, con 3 semillas
mejor = max(resultados, key=lambda n: resultados[n]["final"]["accuracy"])
cfg_final = resultados[mejor]["config"]
print("Configuración elegida en validación:", mejor)
finales = [correr(f"prueba_semilla{s}", cfg_final, ent, pru, semilla=s, medir_cada_epoca=False) for s in (0, 1, 2)]
for m in ("accuracy", "top5", "f1_macro"):
    v = [r["final"][m] for r in finales]
    print(f"prueba sin duplicados | {m}: {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")

# %% [7] Mismo esquema del paper: 5 particiones con todas las filas (tiene duplicados entre partes)
CORRER_CV5 = True
if CORRER_CV5:
    completo = pd.read_csv(f"{CARPETA_DATOS}/CPN120.csv", dtype=str, keep_default_na=False)
    cv = [correr(f"cv5_particion{k}", cfg_final, completo.iloc[i].reset_index(drop=True),
                 completo.iloc[j].reset_index(drop=True), medir_cada_epoca=False)
          for k, (i, j) in enumerate(KFold(5, shuffle=True, random_state=42).split(completo))]
    v = [r["final"]["accuracy"] for r in cv]
    print(f"5 particiones, todas las filas | accuracy: {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")
