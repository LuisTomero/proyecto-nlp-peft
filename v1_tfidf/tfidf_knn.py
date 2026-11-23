"""v1: versión simplificada. Reemplaza Qwen3 + QLoRA por TF-IDF + vecinos más cercanos (kNN).

Los hiperparámetros se eligen solo con validación; la prueba no se usa para elegir.
Uso: python v1_tfidf/tfidf_knn.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

RAIZ = Path(__file__).resolve().parents[1]


def vectorizar(ent, ev, peso_concepto):
    """TF-IDF de trozos de 2 a 5 letras de la respuesta + TF-IDF del concepto (con un peso)."""
    vr = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True)
    vc = TfidfVectorizer()
    a = [vr.fit_transform(ent["Respuesta"]), peso_concepto * vc.fit_transform(ent["Concepto"])]
    b = [vr.transform(ev["Respuesta"]), peso_concepto * vc.transform(ev["Concepto"])]
    return sp.hstack(a).tocsr(), sp.hstack(b).tocsr()


def knn(a, b, codigos, k):
    """Los k vecinos más parecidos votan por su código, con peso igual a su similitud."""
    top5 = []
    for i in range(0, b.shape[0], 400):
        sim = (b[i:i + 400] @ a.T).toarray()
        for fila, vecinos in zip(sim, np.argpartition(-sim, k, axis=1)[:, :k]):
            votos = {}
            for v in vecinos:
                votos[codigos[v]] = votos.get(codigos[v], 0) + fila[v]
            top5.append(sorted(votos, key=votos.get, reverse=True)[:5])
    return top5


def medir(reales, top5):
    top1 = [t[0] for t in top5]
    return {"accuracy": float(np.mean([r == p for r, p in zip(reales, top1)])),
            "top5": float(np.mean([r in t for r, t in zip(reales, top5)])),
            "f1_macro": float(f1_score(reales, top1, labels=np.unique(reales), average="macro", zero_division=0))}


def evaluar(ent, ev, k, peso):
    a, b = vectorizar(ent, ev, peso)
    return medir(ev["Codigo"].values, knn(a, b, ent["Codigo"].values, k))


def main():
    d = pd.read_csv(RAIZ / "datos/cpn120_particion.csv", dtype=str, keep_default_na=False)
    ent, val, pru = (d[d["parte"] == p].reset_index(drop=True) for p in ("entrenamiento", "validacion", "prueba"))

    busqueda = []
    for peso in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
        a, b = vectorizar(ent, val, peso)
        for k in (1, 3, 5, 10):
            m = medir(val["Codigo"].values, knn(a, b, ent["Codigo"].values, k))
            busqueda.append({"k": k, "peso_concepto": peso, **m})
            print("validación", busqueda[-1], flush=True)
    mejor = max(busqueda, key=lambda x: x["accuracy"])
    mejor_sin = max((x for x in busqueda if x["peso_concepto"] == 0), key=lambda x: x["accuracy"])

    res = {"busqueda_validacion": busqueda, "elegido": {"k": mejor["k"], "peso_concepto": mejor["peso_concepto"]}}
    res["prueba_sin_duplicados"] = evaluar(ent, pru, mejor["k"], mejor["peso_concepto"])
    res["prueba_sin_duplicados_solo_respuesta"] = evaluar(ent, pru, mejor_sin["k"], 0.0)

    # Mismo esquema del paper (5 particiones, todas las filas): con fuga, solo para mostrar cuánto infla
    c = pd.read_csv(RAIZ / "datos/CPN120.csv", dtype=str, keep_default_na=False)
    ms = [evaluar(c.iloc[i].reset_index(drop=True), c.iloc[j].reset_index(drop=True), mejor["k"], mejor["peso_concepto"])
          for i, j in KFold(5, shuffle=True, random_state=42).split(c)]
    res["cv5_todas_las_filas_CON_FUGA"] = {m: [float(np.mean([x[m] for x in ms])), float(np.std([x[m] for x in ms], ddof=1))]
                                           for m in ms[0]}
    (RAIZ / "v1_tfidf/resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "busqueda_validacion"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
