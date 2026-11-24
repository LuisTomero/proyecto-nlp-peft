"""Arma v3_qwen_qlora.py a partir del script crudo (v0). Cada cambio está en la lista CAMBIOS, con su motivo."""
from pathlib import Path

aqui = Path(__file__).resolve().parent
raiz = aqui.parent
s = (raiz / "v0_cruda" / "cpn120_qlora.py").read_text(encoding="utf-8")
prep = (raiz / "datos" / "preparar_datos.py").read_text(encoding="utf-8")
prep = prep[prep.index("import re"):prep.index("def main")].replace("CARPETA = Path(__file__).resolve().parent\n", "")
HUELLA = "df77fa49229886e1fa404dfe3592f35a1d2fc31d6172c961fda8cc7c7afae081"

# (texto del script crudo, texto nuevo, cuántas veces aparece)
CAMBIOS = [
    # Ejecución: todo en la carpeta actual (Drive ya montado)
    ("USAR_DRIVE = True ", "USAR_DRIVE = False ", 1),
    ('DIR_RESULTADOS = "resultados_cpn120"\n', 'DIR_RESULTADOS = "resultados_v3"\n', 1),
    # Una sola partición fija (entrenamiento / validación / prueba) en vez de 5
    ("N_PARTICIONES = 5            # igual que el sistema de referencia", "N_PARTICIONES = 1            # partición fija, sin duplicados", 1),
    # Sin duplicados: se usa la misma partición que v1 y v2, y se verifica que sea idéntica
    ("df = df[~vacias].reset_index(drop=True)\n",
     "df = df[~vacias].reset_index(drop=True)\n\n"
     "# Celda copiada de datos/preparar_datos.py\nimport hashlib\n" + prep +
     '\ndf = particionar(df_crudo)[["Concepto", "Respuesta", "Codigo", "parte"]]\n'
     'assert hashlib.sha256(df.to_csv(index=False, lineterminator="\\n").encode("utf-8")).hexdigest() == "' + HUELLA + '"\n', 1),
    ('    entrenamiento = df[df["particion"] != k]\n', '    entrenamiento = df[df["particion"] == 1]\n', 2),
    ("    ent, val = train_test_split(entrenamiento, test_size=FRACCION_VALIDACION, random_state=SEMILLA)\n",
     '    ent, val = entrenamiento, df[df["particion"] == 2]\n', 1),
    ('df.loc[df["particion"] != particion_modelo, "clave"]', 'df.loc[df["particion"] == 1, "clave"]', 1),
    # Corrige el error de v0: con True, la evaluación se recorre en otro orden y las predicciones quedan desalineadas
    ("group_by_length=True,", "group_by_length=False,", 1),
    # 3 épocas: 1.353 pasos, parecido a los 1.514 de v0, que tenía más filas
    ("NUM_EPOCAS = 2\n", "NUM_EPOCAS = 3\n", 1),
]
for viejo, nuevo, veces in CAMBIOS:
    assert s.count(viejo) == veces, (s.count(viejo), viejo)
    s = s.replace(viejo, nuevo)

# La partición fija reemplaza al bloque de validación cruzada (particiones.csv)
ini = s.index("# Validación cruzada estratificada de 5 particiones.")
fin = s.index("# Diagnóstico por partición")
s = (s[:ini] + "# Partición fija: 0 = prueba, 1 = entrenamiento, 2 = validación\n"
     'df["particion"] = df["parte"].map({"prueba": 0, "entrenamiento": 1, "validacion": 2})\n\n' + s[fin:])
s = s.replace('    ent, pru = df[df["particion"] != k], df[df["particion"] == k]\n',
              '    ent, pru = df[df["particion"] == 1], df[df["particion"] == k]\n')
(aqui / "v3_qwen_qlora.py").write_text(s, encoding="utf-8", newline="\n")
print("v3_qwen_qlora.py listo")
