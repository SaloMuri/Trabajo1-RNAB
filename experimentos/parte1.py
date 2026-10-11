"""Parte 1: descenso por gradiente vs GA vs PSO vs DE, contando evaluaciones.

Todos los parametros estan en un archivo YAML (por defecto configs/parte1.yaml):

    python -m experimentos.parte1                      # o:  python main.py
    python -m experimentos.parte1 configs/demo.yaml    # version rapida

Semillas: para cada (funcion, dimension, metodo) las semillas de las corridas salen de
SeedSequence([semilla, indice_funcion, n, indice_metodo]).spawn(corridas), asi que las
corridas son independientes y se pueden repetir.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from src.estadisticas import resumir
from src.funciones import FUNCIONES
from src.metodos import METODOS, crear_problema, ejecutar_metodo

FUNCIONES_ORDEN = list(FUNCIONES)       # el orden fija el indice de cada funcion en las semillas


def cargar_config(ruta="configs/parte1.yaml"):
    with open(ruta, encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)
    validar_config(cfg)
    return cfg


def validar_config(cfg):
    """Falla con un mensaje claro si el YAML tiene un nombre mal escrito."""
    for funcion in cfg["funciones"]:
        if funcion not in FUNCIONES:
            raise ValueError(f"Funcion desconocida {funcion!r}. Opciones: {FUNCIONES_ORDEN}")
        if funcion not in cfg["umbral_exito"]:
            raise ValueError(f"Falta umbral_exito para {funcion!r}")
    for metodo in cfg["metodos"]:
        if metodo not in METODOS:
            raise ValueError(f"Metodo desconocido {metodo!r}. Opciones: {METODOS}")
        if metodo not in cfg:
            raise ValueError(f"Falta la seccion de parametros '{metodo}:' en el YAML")


def correr_una(metodo, funcion, n, rng, cfg):
    """Una corrida independiente. Devuelve un dict con los resultados."""
    problema = crear_problema(funcion, n, cfg["presupuesto_evaluaciones"], cfg.get("costo_gradiente"))
    resultado = ejecutar_metodo(metodo, problema, rng, FUNCIONES[funcion]["dominio"], cfg[metodo])
    return dict(
        f=resultado["f"],
        n_f=problema.n_f,
        n_grad=problema.n_grad,
        total=problema.total_equivalente,
        divergio=resultado["divergio"],
        motivo=resultado["motivo"],
        iteraciones=resultado["iteraciones"],
    )


def semillas_de_las_corridas(cfg, funcion, n, metodo):
    clave = [cfg["semilla"], FUNCIONES_ORDEN.index(funcion), n, METODOS.index(metodo)]
    return np.random.SeedSequence(clave).spawn(cfg["corridas"])


def ejecutar_parte1(cfg):
    """Corre todas las combinaciones funcion x dimension x metodo y guarda los resultados."""
    inicio = time.time()
    filas_corridas, filas_resumen = [], []

    for funcion in cfg["funciones"]:
        for n in cfg["dimensiones"]:
            for metodo in cfg["metodos"]:
                resultados = []
                for k, semilla in enumerate(semillas_de_las_corridas(cfg, funcion, n, metodo)):
                    fila = correr_una(metodo, funcion, n, np.random.default_rng(semilla), cfg)
                    resultados.append(fila)
                    filas_corridas.append(dict(funcion=funcion, n=n, metodo=metodo, corrida=k, **fila))

                resumen = resumir(resultados, FUNCIONES[funcion]["f_opt"], cfg["umbral_exito"][funcion])
                filas_resumen.append(dict(funcion=funcion, n=n, metodo=metodo, **resumen))
                print(f"{funcion:10s} n={n} {metodo:3s}  media={resumen['media']:.3e}  "
                      f"exito={resumen['tasa_exito']:.2f}  total_eq={resumen['total_eq_media']:.0f}",
                      flush=True)

    guardar_tablas(pd.DataFrame(filas_corridas), pd.DataFrame(filas_resumen), cfg)
    print(f"\nListo en {time.time() - inicio:.1f} s. Resultados en {cfg['carpeta_resultados']}/")
    return pd.DataFrame(filas_resumen)


def guardar_tablas(df_corridas, df_resumen, cfg):
    carpeta = cfg["carpeta_resultados"]
    os.makedirs(carpeta, exist_ok=True)
    df_corridas.to_csv(os.path.join(carpeta, "corridas.csv"), index=False)
    df_resumen.to_csv(os.path.join(carpeta, "resumen.csv"), index=False)
    with open(os.path.join(carpeta, "config_usada.yaml"), "w", encoding="utf-8") as archivo:
        yaml.safe_dump(cfg, archivo, sort_keys=False, allow_unicode=True)
    escribir_markdown(df_resumen, os.path.join(carpeta, "resumen.md"))


def escribir_markdown(df, ruta):
    columnas = list(df.columns)
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("| " + " | ".join(columnas) + " |\n")
        archivo.write("|" + "---|" * len(columnas) + "\n")
        for _, fila in df.iterrows():
            celdas = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in fila]
            archivo.write("| " + " | ".join(celdas) + " |\n")


if __name__ == "__main__":
    ejecutar_parte1(cargar_config(sys.argv[1] if len(sys.argv) > 1 else "configs/parte1.yaml"))
