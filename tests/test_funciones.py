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
@pytest.mark.parametrize("n", [2, 3])
def test_gradiente_vectorizado_igual_a_puntual(nombre, n):
    """Regresion: rosenbrock_grad devolvia valores incorrectos (sin error) con poblaciones (m, n)."""
    d = FUNCIONES[nombre]
    lo, hi = d["dominio"]
    X = np.random.default_rng(1).uniform(lo, hi, size=(7, n))
    por_filas = np.array([d["grad"](x) for x in X])
    assert np.allclose(d["grad"](X), por_filas)
    assert np.allclose(d["f"](X), [d["f"](x) for x in X])


@pytest.mark.parametrize("nombre", list(FUNCIONES))
def test_valor_en_el_optimo(nombre):
    d = FUNCIONES[nombre]
    x = np.full(3, d["x_opt"])
    assert abs(d["f"](x) - d["f_opt"]) < 1e-12


@pytest.mark.parametrize("nombre", list(FUNCIONES))
@pytest.mark.parametrize("n", [2, 3])
def test_ningun_punto_baja_del_minimo(nombre, n):
    d = FUNCIONES[nombre]
    lo, hi = d["dominio"]
    X = np.random.default_rng(2).uniform(lo, hi, size=(20000, n))
    assert np.all(d["f"](X) >= d["f_opt"] - 1e-12)


@pytest.mark.parametrize("nombre", list(FUNCIONES))
def test_gradiente_nulo_en_el_optimo(nombre):
    d = FUNCIONES[nombre]
    assert np.allclose(d["grad"](np.full(3, d["x_opt"])), 0.0, atol=1e-9)


def test_conteo():
    d = FUNCIONES["rastrigin"]
    p = Problema(d["f"], d["grad"], n=2)
    p.f(np.zeros(2))                 # 1
    p.f(np.zeros((10, 2)))           # +10 (poblacion)
    p.grad(np.ones(2))               # 1 de grad
    assert p.n_f == 11 and p.n_grad == 1
    assert p.total_equivalente == 11 + 4 * 1   # 2n = 4
    # El gradiente numérico realiza 2n evaluaciones reales de f.
    q = Problema(d["f"], None, n=2)
    q.grad(np.ones(2))
    assert q.n_f == 4 and q.n_grad == 1
    assert q.total_equivalente == 4
