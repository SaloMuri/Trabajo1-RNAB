"""Punto de entrada unico de la Parte 1.

    python main.py                      # usa configs/parte1.yaml
    python main.py configs/demo.yaml    # version rapida

Todos los parametros (semilla, corridas, presupuesto, hiperparametros, GIFs) se cambian
en el archivo YAML, no aqui.
"""
import sys

from experimentos.gifs import generar_gifs
from experimentos.parte1 import cargar_config, ejecutar_parte1


def main():
    ruta_config = sys.argv[1] if len(sys.argv) > 1 else "configs/parte1.yaml"
    cfg = cargar_config(ruta_config)
    print(f"Configuracion: {ruta_config}\n")
    ejecutar_parte1(cfg)
    print("\nGenerando GIFs...")
    generar_gifs(cfg)
    print(f"\nListo. Tablas en {cfg['carpeta_resultados']}/ y GIFs en {cfg['gifs']['carpeta']}/")


if __name__ == "__main__":
    main()
