"""Genera las graficas estadisticas de la Parte 1 a partir de los CSV.

Uso:
    python -m experimentos.graficas
    python -m experimentos.graficas configs/demo.yaml

No vuelve a ejecutar los algoritmos: utiliza corridas.csv y resumen.csv ya guardados.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimentos.parte1 import cargar_config


METODOS_NOMBRE = {
    "gd": "Descenso por gradiente",
    "ga": "Algoritmo genético",
    "pso": "PSO",
    "de": "Evolución diferencial",
}
ORDEN_METODOS = ["gd", "ga", "pso", "de"]


def generar_graficas(cfg):
    """Lee los CSV del experimento y guarda seis PNG dentro de resultados/parte1/figuras."""
    carpeta = cfg["carpeta_resultados"]
    ruta_corridas = os.path.join(carpeta, "corridas.csv")
    ruta_resumen = os.path.join(carpeta, "resumen.csv")
    if not os.path.isfile(ruta_corridas) or not os.path.isfile(ruta_resumen):
        raise FileNotFoundError(
            f"No encuentro {ruta_corridas} o {ruta_resumen}. "
            "Ejecuta primero python main.py para producir los CSV."
        )

    corridas = pd.read_csv(ruta_corridas)
    resumen = pd.read_csv(ruta_resumen)
    carpeta_figuras = os.path.join(carpeta, "figuras")
    os.makedirs(carpeta_figuras, exist_ok=True)

    # Cuatro diagramas de caja: uno por función y dimensión.
    for funcion in cfg["funciones"]:
        for n in cfg["dimensiones"]:
            datos = corridas[(corridas["funcion"] == funcion) & (corridas["n"] == n)]
            metodos = [m for m in ORDEN_METODOS if m in datos["metodo"].unique()]
            valores = [datos.loc[datos["metodo"] == m, "f"].dropna().to_numpy() for m in metodos]

            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            ax.boxplot(valores, tick_labels=[METODOS_NOMBRE[m] for m in metodos], showfliers=True)
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.set_ylabel("Valor final de la función objetivo f(x) (escala symlog)")
            ax.set_title(f"Distribución de resultados: {funcion.capitalize()}, {n}D")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelrotation=12)
            fig.tight_layout()
            nombre = f"distribucion_{funcion}_{n}D.png"
            fig.savefig(os.path.join(carpeta_figuras, nombre), dpi=160, bbox_inches="tight")
            plt.close(fig)

    # Dos diagramas de barras: tasa de éxito para ambas dimensiones y los cuatro métodos.
    for funcion in cfg["funciones"]:
        datos = resumen[resumen["funcion"] == funcion]
        ns = sorted(datos["n"].unique())
        metodos = [m for m in ORDEN_METODOS if m in datos["metodo"].unique()]
        x = np.arange(len(metodos))
        ancho = 0.8 / max(1, len(ns))
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for j, n in enumerate(ns):
            sub = datos[datos["n"] == n].set_index("metodo")
            tasas = [float(sub.loc[m, "tasa_exito"]) * 100 if m in sub.index else 0 for m in metodos]
            ax.bar(x + (j - (len(ns) - 1) / 2) * ancho, tasas, width=ancho,
                   label=f"{n}D")
        ax.set_xticks(x)
        ax.set_xticklabels([METODOS_NOMBRE[m] for m in metodos], rotation=12, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Corridas exitosas (%)")
        ax.set_title(f"Tasa de éxito: {funcion.capitalize()}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Dimensión")
        fig.tight_layout()
        nombre = f"tasa_exito_{funcion}.png"
        fig.savefig(os.path.join(carpeta_figuras, nombre), dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"Se generaron 6 gráficas en: {carpeta_figuras}/")
    return carpeta_figuras


if __name__ == "__main__":
    ruta_config = sys.argv[1] if len(sys.argv) > 1 else "configs/parte1.yaml"
    generar_graficas(cargar_config(ruta_config))
