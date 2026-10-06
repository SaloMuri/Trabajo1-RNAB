"""Verifica el gradiente analitico contra diferencias finitas y el conteo."""
import numpy as np
import pytest
from src.funciones import FUNCIONES
from src.contador import Problema


@pytest.mark.parametrize("nombre", list(FUNCIONES))
@pytest.mark.parametrize("n", [2, 3])
def test_gradiente_vs_diferencias_finitas(nombre, n):
    d = FUNCIONES[nombre]
    rng = np.random.default_rng(0)
    lo, hi = d["dominio"]
    for _ in range(20):
        x = rng.uniform(lo, hi, size=n)
        num = Problema(d["f"], None, n)._grad_numerico(x)
        assert np.allclose(d["grad"](x), num, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("nombre", list(FUNCIONES))
def test_valor_en_el_optimo(nombre):
    d = FUNCIONES[nombre]
    x = np.full(3, d["x_opt"])
    assert abs(d["f"](x) - d["f_opt"]) < 1e-12


def test_conteo():
    d = FUNCIONES["rastrigin"]
    p = Problema(d["f"], d["grad"], n=2)
    p.f(np.zeros(2))                 # 1
    p.f(np.zeros((10, 2)))           # +10 (poblacion)
    p.grad(np.ones(2))               # 1 de grad
    assert p.n_f == 11 and p.n_grad == 1
    assert p.total_equivalente == 11 + 4 * 1   # 2n = 4
    # gradiente numerico no debe inflar n_f
    q = Problema(d["f"], None, n=2)
    q.grad(np.ones(2))
    assert q.n_f == 0 and q.n_grad == 1
