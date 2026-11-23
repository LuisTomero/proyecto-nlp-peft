"""Arma v0_cruda_kaggle.ipynb desde el script crudo. Los cambios respecto del original son solo estos:
  - una celda inicial que deja una sola GPU visible y copia el CSV a la carpeta de trabajo;
  - USAR_DRIVE = False, porque Kaggle no usa Google Drive;
  - PARTICIONES_A_EJECUTAR = [0], por tiempo.
"""
import json
import re
from pathlib import Path

aqui = Path(__file__).resolve().parent
crudo = (aqui.parent / "v0_cruda" / "cpn120_qlora.py").read_text(encoding="utf-8")

INICIAL = '''# Celda agregada para Kaggle (no es parte de la versión cruda)
import glob
import os
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Kaggle ofrece dos T4; el script está hecho para una
RUTAS = glob.glob("/kaggle/input/**/CPN120.csv", recursive=True)
assert RUTAS, "Agrega al notebook el dataset con CPN120.csv (Add Input)."
shutil.copy(RUTAS[0], "CPN120.csv")
print("CSV copiado desde", RUTAS[0])
'''
CAMBIOS = [
    ("USAR_DRIVE = True            # guarda resultados en Drive para poder retomar si se corta la sesión",
     "USAR_DRIVE = False           # CAMBIO respecto de la versión cruda: Kaggle no usa Drive"),
    ("PARTICIONES_A_EJECUTAR = list(range(N_PARTICIONES))  # p. ej. [0] para una prueba rápida",
     "PARTICIONES_A_EJECUTAR = [0]  # CAMBIO respecto de la versión cruda: solo la partición 0"),
]
for viejo, nuevo in CAMBIOS:
    assert crudo.count(viejo) == 1, viejo
    crudo = crudo.replace(viejo, nuevo)

partes = [INICIAL] + re.split(r"(?m)^(?=# %% )", crudo)
celdas = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
           "source": p.rstrip("\n").splitlines(True)} for p in partes]
nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
(aqui / "v0_cruda_kaggle.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(celdas), "celdas")
