"""Pruebas para comprobar que las gráficas se pueden regenerar desde los CSV."""
import os
import shutil

import pandas as pd

from experimentos.graficas import generar_graficas


def test_graficas_se_generan_desde_los_csv(tmp_path):
    origen = "resultados/parte1"
    corridas = pd.read_csv(os.path.join(origen, "corridas.csv"))
    resumen = pd.read_csv(os.path.join(origen, "resumen.csv"))
    corridas.to_csv(tmp_path / "corridas.csv", index=False)
    resumen.to_csv(tmp_path / "resumen.csv", index=False)

    cfg = {
        "carpeta_resultados": str(tmp_path),
        "funciones": ["rosenbrock", "rastrigin"],
        "dimensiones": [2, 3],
    }
    carpeta = generar_graficas(cfg)
    archivos = os.listdir(carpeta)
    assert len([nombre for nombre in archivos if nombre.endswith(".png")]) == 6
    assert os.path.exists(os.path.join(carpeta, "distribucion_rosenbrock_2D.png"))
    assert os.path.exists(os.path.join(carpeta, "tasa_exito_rastrigin.png"))
