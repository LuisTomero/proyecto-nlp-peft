"""Arma evaluar_alineado.py desde el script crudo: carga el modelo ya entrenado de la partición 0 y lo mide en el
orden correcto, sin reentrenar. Necesita resultados_cpn120/ (adaptador_particion0 y particiones.csv) en la carpeta actual.

En Colab: copiar resultados_cpn120/particiones.csv a resultados_cpn120_corregido/ y correr python evaluar_alineado.py
"""
from pathlib import Path

aqui = Path(__file__).resolve().parent
s = (aqui.parent / "v0_cruda" / "cpn120_qlora.py").read_text(encoding="utf-8")

CAMBIOS = [
    ("USAR_DRIVE = True ", "USAR_DRIVE = False "),
    ('DIR_RESULTADOS = "resultados_cpn120"\n', 'DIR_RESULTADOS = "resultados_cpn120_corregido"\n'),
    ("PARTICIONES_A_EJECUTAR = list(range(N_PARTICIONES))  # p. ej. [0] para una prueba rápida\n",
     "PARTICIONES_A_EJECUTAR = [0]\n"),
    ("GUARDAR_ADAPTADORES = True ", "GUARDAR_ADAPTADORES = False "),
    # Usa el modelo ya entrenado en vez de uno nuevo, y no entrena
    ("\nmodelo = crear_modelo()\n",
     '\nmodelo = PeftModel.from_pretrained(cargar_modelo_base(), "resultados_cpn120/adaptador_particion0")\n'),
    ("    trainer.train()\n", "    pass\n"),
    # La corrección: con True, la evaluación se recorre en otro orden
    ("group_by_length=True,", "group_by_length=False,"),
]
for viejo, nuevo in CAMBIOS:
    assert s.count(viejo) == 1, (s.count(viejo), viejo)
    s = s.replace(viejo, nuevo)
(aqui / "evaluar_alineado.py").write_text(s, encoding="utf-8", newline="\n")
print("evaluar_alineado.py listo")
