"""Contador de evaluaciones.

Los algoritmos no cuentan nada por su cuenta: todos reciben un objeto `Problema` y lo
llaman. Asi el conteo queda en un solo lugar.

  * n_f    : numero de evaluaciones de f (una poblacion de m puntos cuenta m).
  * n_grad : numero de evaluaciones del gradiente.
  * Un gradiente "vale" `costo_grad_en_f` evaluaciones de f. Por defecto 2n, que es lo que
    costaria aproximar el gradiente con diferencias centrales en dimension n.
  * total_equivalente = n_f + costo_grad_en_f * n_grad  (esto se compara con el presupuesto).
  * evals_exito : evaluaciones equivalentes que se habian gastado cuando el mejor valor de f 
  bajo por primera vez de `meta` (None si nunca llego). Mide que tan rapido llega cada metodo.
  * historial : lista de (total_equivalente, mejor_f), una entrada cada vez que el mejor valor
  mejora. Sirve para graficar costo vs evaluaciones de cualquier metodo sin tocar los metodos.
"""

import numpy as np


class Problema:
    def __init__(self, f, grad=None, n=2, costo_grad_en_f=None, max_evals=None, meta = None):
        self._f = f
        self._grad = grad
        self.n = n
        self.costo_grad_en_f = 2 * n if costo_grad_en_f is None else costo_grad_en_f
        self.max_evals = max_evals
        self.n_f = 0
        self.n_grad = 0
        self.mejor_f = np.inf       # mejor valor de f visto hasta ahora
        self.mejor_x = None
        self.meta = meta            # valor de f que cuenta como exito (f_optimo + umbral); None = no medir
        self.evals_exito = None     # se llena la primera vez que mejor_f < meta
        self.historial = []         # (total_equivalente, mejor_f) cada vez que mejor_f mejora
        

    def f(self, x):
        """Evalua f en un punto (n,) o en una poblacion (m, n) y actualiza el mejor valor."""
        x = np.asarray(x, dtype=float)
        mejor_antes = self.mejor_f
        if x.ndim == 2:
            self.n_f += x.shape[0]
            valores = self._f(x)
            i = int(np.argmin(valores))
            if valores[i] < self.mejor_f:
                self.mejor_f, self.mejor_x = float(valores[i]), x[i].copy()
        else:
            self.n_f += 1
            valores = self._f(x)
            if valores < self.mejor_f:
                self.mejor_f, self.mejor_x = float(valores), x.copy()
        if self.mejor_f < mejor_antes:
            self.historial.append((self.total_equivalente, self.mejor_f))
        self._revisar_exito()
        return valores
    

    def _revisar_exito(self):
        """Guarda cuantas evaluaciones equivalentes se llevaban al llegar por primera vez a la meta."""
        if self.meta is not None and self.evals_exito is None and self.mejor_f < self.meta:
            self.evals_exito = self.total_equivalente
            

    def grad(self, x):
        """Evalua el gradiente (analitico). Si no hay, usa diferencias centrales."""
        x = np.asarray(x, dtype=float)
        self.n_grad += 1
        if self._grad is not None:
            return self._grad(x)
        return self._grad_numerico(x)

    def _grad_numerico(self, x, h=1e-6):
        g = np.zeros_like(x)
        for i in range(self.n):
            e = np.zeros_like(x)
            e[i] = h
            g[i] = (self.f(x + e) - self.f(x - e)) / (2 * h)
        return g

    @property
    def total_equivalente(self):
        if self._grad is None:
            # El gradiente numérico ya cuenta sus evaluaciones en n_f.
            return self.n_f
        return self.n_f + self.costo_grad_en_f * self.n_grad

    def hay_presupuesto(self, costo=1):
        """True si gastar `costo` evaluaciones mas no pasa del presupuesto."""
        return self.max_evals is None or self.total_equivalente + costo <= self.max_evals
