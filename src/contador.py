"""Envoltorio que cuenta evaluaciones de f y de grad f de forma automatica.

Idea: ningun algoritmo cuenta evaluaciones por su cuenta. Todos reciben un
objeto `Problema` y lo llaman; el conteo ocurre aqui, en un unico lugar.
"""
import numpy as np


class Problema:
    def __init__(self, f, grad=None, n=2, costo_grad_en_f=None):
        """
        f      : funcion objetivo. Acepta un vector (n,) o una poblacion (m, n).
        grad   : gradiente analitico (opcional).
        n      : dimension.
        costo_grad_en_f : cuantas evaluaciones de f "vale" una evaluacion del
                 gradiente. Si es None se usa 2n (diferencias centrales).
                 ESTA EQUIVALENCIA HAY QUE JUSTIFICARLA EN EL REPORTE.
        """
        self._f = f
        self._grad = grad
        self.n = n
        self.costo_grad_en_f = 2 * n if costo_grad_en_f is None else costo_grad_en_f
        self.n_f = 0
        self.n_grad = 0

    def f(self, x):
        x = np.asarray(x, dtype=float)
        # una poblacion (m, n) cuenta como m evaluaciones
        self.n_f += x.shape[0] if x.ndim == 2 else 1
        return self._f(x)

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        self.n_grad += 1
        if self._grad is not None:
            return self._grad(x)
        return self._grad_numerico(x)

    def _grad_numerico(self, x, h=1e-6):
        # Diferencias centrales: usa _f directo para NO contar las 2n
        # evaluaciones como evaluaciones de f (ya se cuentan como 1 de grad).
        g = np.zeros_like(x)
        for i in range(self.n):
            e = np.zeros_like(x)
            e[i] = h
            g[i] = (self._f(x + e) - self._f(x - e)) / (2 * h)
        return g

    @property
    def total_equivalente(self):
        """Evaluaciones totales expresadas en unidades de f."""
        return self.n_f + self.costo_grad_en_f * self.n_grad

    def reiniciar(self):
        self.n_f = 0
        self.n_grad = 0
