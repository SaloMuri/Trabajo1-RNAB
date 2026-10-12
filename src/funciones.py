"""Funciones de prueba y sus gradientes.

Cada funcion recibe un punto x de forma (n,) o varios puntos de forma (m, n) y devuelve
el valor de f (un numero, o m numeros). Cada gradiente devuelve algo de la misma forma que x.

Resumen de lo que se usa:
  Rosenbrock : f(x) = sum_{i=1}^{n-1} [ 100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
               dominio [-2.048, 2.048]^n, minimo global f = 0 en (1, ..., 1).
               Tiene un valle curvo y angosto: es facil llegar al valle pero lento avanzar por el.
  Rastrigin  : f(x) = 10 n + sum_{i=1}^{n} [ x_i^2 - 10 cos(2 pi x_i) ]
               dominio [-5.12, 5.12]^n, minimo global f = 0 en (0, ..., 0).
               Tiene muchisimos minimos locales (uno cerca de cada entero).

OJO: dominio y minimo global deben confirmarse con una fuente primaria y citarse en el reporte.
"""
import numpy as np


def rosenbrock(x):
    """Valor de Rosenbrock. Suma sobre pares de coordenadas vecinas (x_i, x_{i+1})."""
    x = np.asarray(x, dtype=float)
    a, b = x[..., :-1], x[..., 1:]          # a = x_i,  b = x_{i+1}
    return np.sum(100.0 * (b - a**2) ** 2 + (1.0 - a) ** 2, axis=-1)


def rosenbrock_grad(x):
    """Gradiente de Rosenbrock (derivado a mano).

    Cada coordenada x_i aparece en dos terminos de la suma: como 'a' (con x_{i+1}) y como
    'b' (con x_{i-1}), por eso se suman dos aportes.
    """
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    a, b = x[..., :-1], x[..., 1:]
    g[..., :-1] += -400.0 * a * (b - a**2) - 2.0 * (1.0 - a)    # aporte de x_i como 'a'
    g[..., 1:] += 200.0 * (b - a**2)                            # aporte de x_{i+1} como 'b'
    return g


def rastrigin(x):
    """Valor de Rastrigin. n es el numero de coordenadas."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def rastrigin_grad(x):
    """Gradiente de Rastrigin: cada coordenada se deriva por separado."""
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


# nombre -> funcion, gradiente, dominio (por coordenada), optimo (por coordenada) y f(optimo)
FUNCIONES = {
    "rosenbrock": dict(f=rosenbrock, grad=rosenbrock_grad,
                       dominio=(-2.048, 2.048), x_opt=1.0, f_opt=0.0),
    "rastrigin": dict(f=rastrigin, grad=rastrigin_grad,
                      dominio=(-5.12, 5.12), x_opt=0.0, f_opt=0.0),
}
