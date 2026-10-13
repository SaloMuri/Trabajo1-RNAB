"""Compara los gradientes analíticos con diferencias centrales en puntos aleatorios.

Ejecutar desde la raíz del proyecto:
    python -m experimentos.auditar_gradientes

Usa 100 puntos por función y dimensión, paso h=1e-6 y semilla fija 2026.
El resultado es una verificación numérica reproducible, no una demostración formal.
"""
import numpy as np
from src.funciones import FUNCIONES


def gradiente_central(f, x, h=1e-6):
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def main():
    rng = np.random.default_rng(2026)
    h = 1e-6
    puntos = 100
    print("Función       n   puntos   h         error absoluto máximo (norma infinito)")
    for nombre, datos in FUNCIONES.items():
        lo, hi = datos["dominio"]
        for n in (2, 3):
            errores = []
            for _ in range(puntos):
                x = rng.uniform(lo, hi, size=n)
                g_analitico = datos["grad"](x)
                g_numerico = gradiente_central(datos["f"], x, h=h)
                errores.append(np.linalg.norm(g_analitico - g_numerico, ord=np.inf))
            print(f"{nombre:13s} {n}   {puntos:3d}     {h:.0e}   {max(errores):.9e}")


if __name__ == "__main__":
    main()
