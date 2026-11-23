"""Convierte beto_lora.py (celdas '# %%') en un notebook para Colab."""
import json
import re
from pathlib import Path

aqui = Path(__file__).resolve().parent
partes = re.split(r"(?m)^(?=# %% )", (aqui / "beto_lora.py").read_text(encoding="utf-8"))
prep = (aqui.parent / "datos" / "preparar_datos.py").read_text(encoding="utf-8")
prep = "# Celda copiada de datos/preparar_datos.py\n" + prep[prep.index("import re"):prep.index("def main")]
prep = prep.replace("CARPETA = Path(__file__).resolve().parent\n", "")
partes.insert(1 if partes[0].strip() else 2, prep)
celdas = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
           "source": p.rstrip("\n").splitlines(True)} for p in partes if p.strip()]
nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}}
(aqui / "beto_lora_colab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(celdas), "celdas")
