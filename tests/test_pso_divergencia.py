
import numpy as np

from src.contador import Problema
from src.pso import pso


def test_pso_detecta_divergencia_y_respeta_presupuesto():
    llamadas = 0

    def funcion(x):
        nonlocal llamadas
        llamadas += 1

        # La poblacion inicial funciona; la siguiente evaluacion falla.
        if llamadas >= 2:
            return np.full(x.shape[0], np.nan)

        return np.sum(x**2, axis=1)

    presupuesto = 25
    poblacion = 10

    problema = Problema(
        funcion,
        n=2,
        max_evals=presupuesto
    )

    resultado = pso(
        problema,
        np.random.default_rng(42),
        lo=-5.0,
        hi=5.0,
        pop=poblacion
    )

    assert resultado["divergio"] is True
    assert resultado["motivo"] == "divergencia"
    assert problema.n_f <= presupuesto
    assert problema.n_f == 2 * poblacion
    assert resultado["iteraciones"] == 0 or resultado["iteraciones"] == 1
    assert resultado["x"] is not None
    assert np.all(np.isfinite(resultado["x"]))
    assert np.isfinite(resultado["f"])
    assert np.isclose(
        resultado["f"],
        np.sum(resultado["x"] ** 2)
    )


def test_pso_detecta_divergencia_en_poblacion_inicial():
    def funcion(x):
        return np.full(x.shape[0], np.nan)

    presupuesto = 20
    poblacion = 10

    problema = Problema(
        funcion,
        n=2,
        max_evals=presupuesto
    )

    resultado = pso(
        problema,
        np.random.default_rng(42),
        lo=-5.0,
        hi=5.0,
        pop=poblacion
    )

    assert resultado["divergio"] is True
    assert resultado["motivo"] == "divergencia"
    assert problema.n_f == poblacion
    assert problema.n_f <= presupuesto


def test_pso_presupuesto_menor_que_poblacion():
    def funcion(x):
        return np.sum(x**2, axis=1)

    presupuesto = 5
    poblacion = 10

    problema = Problema(
        funcion,
        n=2,
        max_evals=presupuesto
    )

    resultado = pso(
        problema,
        np.random.default_rng(42),
        lo=-5.0,
        hi=5.0,
        pop=poblacion
    )

    assert problema.n_f == presupuesto
    assert problema.n_f <= presupuesto

def test_pso_no_excede_presupuesto_si_no_alcanza_para_otra_iteracion():
    def funcion(x):
        return np.sum(x**2, axis=1)

    presupuesto = 23
    poblacion = 10

    problema = Problema(
        funcion,
        n=2,
        max_evals=presupuesto
    )

    resultado = pso(
        problema,
        np.random.default_rng(42),
        lo=-5.0,
        hi=5.0,
        pop=poblacion
    )

    assert problema.n_f == 20
    assert problema.n_f <= presupuesto
    assert resultado["iteraciones"] == 1
    assert resultado["motivo"] == "presupuesto"


def test_pso_detecta_menos_infinito_y_conserva_mejor_solucion():
    llamadas = 0

    def funcion_objetivo(x):
        nonlocal llamadas
        llamadas += 1
        valores = np.sum(x**2, axis=1)

        # La población inicial es válida; en la segunda evaluación
        # aparece un valor -inf.
        if llamadas == 2:
            valores[0] = -np.inf

        return valores

    presupuesto = 25
    p = Problema(funcion_objetivo, n=2, max_evals=presupuesto)
    rng = np.random.default_rng(0)

    resultado = pso(p, rng, lo=-5, hi=5, pop=10)

    assert resultado["divergio"] is True
    assert resultado["motivo"] == "divergencia"
    assert p.n_f <= presupuesto

    # La solución devuelta debe seguir siendo válida.
    assert resultado["x"] is not None
    assert np.all(np.isfinite(resultado["x"]))
    assert np.isfinite(resultado["f"])
    assert np.isclose(
        resultado["f"],
        np.sum(resultado["x"]**2)
    )


def test_pso_diverge_en_poblacion_inicial_sin_soluciones_validas():
    def funcion_objetivo(x):
        return np.full(x.shape[0], -np.inf)

    presupuesto = 20
    p = Problema(funcion_objetivo, n=2, max_evals=presupuesto)
    rng = np.random.default_rng(0)

    resultado = pso(p, rng, lo=-5, hi=5, pop=10)

    assert resultado["divergio"] is True
    assert resultado["motivo"] == "divergencia"
    assert resultado["x"] is None
    assert resultado["f"] == np.inf
    assert p.n_f <= presupuesto
    assert p.n_f == 10
