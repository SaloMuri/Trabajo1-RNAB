"""Genera los GIFs definidos en la seccion `gifs:` del YAML (siempre en 2D).

    python -m experimentos.gifs                      # usa configs/parte1.yaml
    python -m experimentos.gifs configs/demo.yaml
Para agregar o quitar GIFs, edite `gifs.lista` en el YAML.
"""
import os
import sys

import numpy as np

from experimentos.parte1 import cargar_config
from src.animacion import crear_gif
from src.funciones import FUNCIONES
from src.metodos import crear_problema, ejecutar_metodo


def generar_gif(cfg, funcion, metodo, semilla, presupuesto):
    """Corre UN metodo en 2D guardando su trayectoria y la convierte en un GIF."""
    n = 2
    rng = np.random.default_rng(semilla)
    problema = crear_problema(funcion, n, presupuesto, cfg.get("costo_gradiente"))
    resultado = ejecutar_metodo(metodo, problema, rng, FUNCIONES[funcion]["dominio"],
                                cfg[metodo], guardar_trayectoria=True)

    g = cfg["gifs"]
    os.makedirs(g["carpeta"], exist_ok=True)
    ruta = os.path.join(g["carpeta"], f"{funcion}_{metodo}_seed{semilla}.gif")
    titulo = f"{metodo.upper()} sobre {funcion} (mejor f final = {resultado['f']:.2e})"
    crear_gif(funcion, resultado["trayectoria"], ruta, titulo, es_poblacion=(metodo != "gd"),
              max_cuadros=g.get("max_cuadros", 60), fps=g.get("fps", 10))
    print(f"GIF: {ruta} | n_f={problema.n_f} n_grad={problema.n_grad} "
          f"total_eq={problema.total_equivalente} mejor_f={resultado['f']:.3e}")


def generar_gifs(cfg):
    """Genera todos los GIFs de la lista `gifs.lista` del YAML."""
    g = cfg["gifs"]
    for item in g["lista"]:
        generar_gif(cfg, item["funcion"], item["metodo"],
                    semilla=item.get("semilla", g["semilla"]),
                    presupuesto=item.get("presupuesto", g["presupuesto_evaluaciones"]))


if __name__ == "__main__":
    generar_gifs(cargar_config(sys.argv[1] if len(sys.argv) > 1 else "configs/parte1.yaml"))
