"""Funciones de prueba con gradiente analitico.

OJO: dominio y minimo global de cada funcion deben verificarse contra una
fuente primaria (y citarse en el reporte). No confiar solo en la IA.
Todas aceptan x de forma (n,) o (m, n).
"""
import numpy as np


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    a, b = x[..., :-1], x[..., 1:]
    return np.sum(100.0 * (b - a**2) ** 2 + (1.0 - a) ** 2, axis=-1)


def rosenbrock_grad(x):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    a, b = x[:-1], x[1:]
    g[:-1] += -400.0 * a * (b - a**2) - 2.0 * (1.0 - a)
    g[1:] += 200.0 * (b - a**2)
    return g


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def rastrigin_grad(x):
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


# nombre -> (f, grad, dominio por coordenada, optimo por coordenada, f(optimo))
# Valores a confirmar con fuente primaria antes de citarlos.
FUNCIONES = {
    "rosenbrock": dict(f=rosenbrock, grad=rosenbrock_grad,
                       dominio=(-2.048, 2.048), x_opt=1.0, f_opt=0.0),
    "rastrigin": dict(f=rastrigin, grad=rastrigin_grad,
                      dominio=(-5.12, 5.12), x_opt=0.0, f_opt=0.0),
}
