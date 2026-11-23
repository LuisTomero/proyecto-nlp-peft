"""Quita los duplicados de CPN120 y fija una partición entrenamiento/validación/prueba.

Uso: python datos/preparar_datos.py   (crea datos/cpn120_particion.csv)
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CARPETA = Path(__file__).resolve().parent
SEMILLA = 42


def normalizar(texto):
    """Minúsculas, sin tildes (conserva la ñ), sin signos y sin espacios repetidos."""
    texto = str(texto).lower().replace("ñ", "\0")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c)).replace("\0", "ñ")
    texto = re.sub(r"[^a-z0-9ñ ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def quitar_duplicados(df):
    """Deja una fila por par concepto + respuesta normalizada, con su código más frecuente."""
    df = df.copy()
    df["clave"] = df["Concepto"].map(normalizar) + " || " + df["Respuesta"].map(normalizar)
    conteo = df.groupby(["clave", "Codigo"]).size().reset_index(name="n")
    # Si hay empate entre códigos, se toma el primero en orden alfabético (siempre el mismo)
    conteo = conteo.sort_values(["clave", "n", "Codigo"], ascending=[True, False, True])
    codigo = conteo.drop_duplicates("clave").set_index("clave")["Codigo"]
    unicos = df.drop_duplicates("clave").sort_values("clave").reset_index(drop=True)
    unicos["Codigo"] = unicos["clave"].map(codigo)
    return unicos


def main():
    df = pd.read_csv(CARPETA / "CPN120.csv", dtype=str, keep_default_na=False)
    unicos = quitar_duplicados(df)
    # 70 % entrenamiento, 15 % validación, 15 % prueba
    resto, prueba = train_test_split(unicos.index, test_size=0.15, random_state=SEMILLA)
    ent, val = train_test_split(resto, test_size=0.15 / 0.85, random_state=SEMILLA)
    unicos["parte"] = ""
    unicos.loc[ent, "parte"] = "entrenamiento"
    unicos.loc[val, "parte"] = "validacion"
    unicos.loc[prueba, "parte"] = "prueba"
    unicos[["Concepto", "Respuesta", "Codigo", "parte"]].to_csv(
        CARPETA / "cpn120_particion.csv", index=False, lineterminator="\n")

    claves = {p: set(unicos.loc[unicos["parte"] == p, "clave"]) for p in ("entrenamiento", "validacion", "prueba")}
    assert not claves["entrenamiento"] & claves["prueba"] and not claves["validacion"] & claves["prueba"]
    assert not claves["entrenamiento"] & claves["validacion"]
    print(f"Filas originales: {len(df)} | sin duplicados: {len(unicos)} | códigos: {unicos['Codigo'].nunique()}")
    print(unicos["parte"].value_counts().to_string())
    vistos = set(unicos.loc[unicos["parte"] == "entrenamiento", "Codigo"])
    for p in ("validacion", "prueba"):
        x = unicos.loc[unicos["parte"] == p, "Codigo"]
        print(f"{p}: {(~x.isin(vistos)).mean():.1%} tiene un código que no está en entrenamiento")


if __name__ == "__main__":
    main()
