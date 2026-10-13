"""Crea un problema y ejecuta un metodo por su nombre.

Nombres: "gd" (descenso por gradiente), "ga" (algoritmo genetico), "pso", "de".
"""
from src.contador import Problema
from src.funciones import FUNCIONES
from src.descenso import descenso_gradiente
from src.ga import algoritmo_genetico
from src.pso import pso
from src.de import evolucion_diferencial

METODOS = ["gd", "ga", "pso", "de"]     # el orden fija el indice de cada metodo en las semillas


def crear_problema(funcion, n, presupuesto, costo_gradiente=None, meta=None):
    """Problema con conteo de evaluaciones. costo_gradiente=None -> 2n. meta: f que cuenta como exito."""
    d = FUNCIONES[funcion]
    return Problema(d["f"], d["grad"], n=n, costo_grad_en_f=costo_gradiente,
                    max_evals=presupuesto, meta=meta)


def ejecutar_metodo(metodo, problema, rng, dominio, parametros, guardar_trayectoria=False):
    """Corre un metodo. `parametros` es la seccion del metodo en el YAML."""
    lo, hi = dominio
    parametros = {k: v for k, v in parametros.items() if v is not None}

    if metodo == "gd":
        x0 = rng.uniform(lo, hi, size=problema.n)       # condicion inicial aleatoria
        return descenso_gradiente(problema, x0, guardar_trayectoria=guardar_trayectoria, limites=dominio, **parametros)
    if metodo == "ga":
        return algoritmo_genetico(problema, rng, lo, hi, guardar_trayectoria=guardar_trayectoria, **parametros)
    if metodo == "pso":
        return pso(problema, rng, lo, hi, guardar_trayectoria=guardar_trayectoria, **parametros)
    if metodo == "de":
        return evolucion_diferencial(problema, rng, lo, hi, guardar_trayectoria=guardar_trayectoria, **parametros)
    raise ValueError(f"Metodo desconocido: {metodo!r}. Opciones: {METODOS}")
