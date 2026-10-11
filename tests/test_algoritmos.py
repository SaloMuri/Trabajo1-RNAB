"""Conteo de evaluaciones, presupuesto, reproducibilidad y sanidad de los algoritmos."""
import numpy as np
from src.contador import Problema
from src.funciones import FUNCIONES
from src.descenso import descenso_gradiente
from src.pso import pso
from src.de import evolucion_diferencial
from src.ga import algoritmo_genetico


def _problema(nombre, n, max_evals):
    d = FUNCIONES[nombre]
    return Problema(d["f"], d["grad"], n=n, max_evals=max_evals), d["dominio"]


def test_de_conteo_exacto():
    pop, gens = 20, 50
    p, (lo, hi) = _problema("rastrigin", 2, pop * (gens + 1))
    r = evolucion_diferencial(p, np.random.default_rng(0), lo, hi, pop=pop)
    assert p.n_f == pop * (gens + 1) and p.n_grad == 0 and r["iteraciones"] == gens


def test_pso_conteo_exacto():
    pop, iters = 15, 40
    p, (lo, hi) = _problema("rosenbrock", 3, pop * (iters + 1))
    r = pso(p, np.random.default_rng(0), lo, hi, pop=pop)
    assert p.n_f == pop * (iters + 1) and r["iteraciones"] == iters


def test_ga_conteo_exacto():
    pop, elite, gens = 30, 2, 25
    p, (lo, hi) = _problema("rastrigin", 2, pop + (pop - elite) * gens)
    r = algoritmo_genetico(p, np.random.default_rng(0), lo, hi, pop=pop, elite=elite)
    assert p.n_f == pop + (pop - elite) * gens and r["iteraciones"] == gens


def test_gd_conteo_paso_fijo():
    # cada iteracion completa = 1 gradiente + 1 evaluacion de f; mas 1 f inicial.
    # Si se para por tolerancia del gradiente, el ultimo gradiente no va seguido de f.
    p, (lo, hi) = _problema("rastrigin", 2, 10**6)
    r = descenso_gradiente(p, np.array([0.3, -0.2]), lr=1e-3, tol_grad=1e-10)
    completas = p.n_f - 1
    assert p.n_grad == completas + (1 if r["motivo"] == "gradiente" else 0)
    assert p.total_equivalente == p.n_f + 4 * p.n_grad



def test_ningun_metodo_excede_el_presupuesto():
    for nombre in FUNCIONES:
        for n in (2, 3):
            B = 3000
            for metodo in ("gd", "ga", "pso", "de"):
                p, (lo, hi) = _problema(nombre, n, B)
                rng = np.random.default_rng(1)
                if metodo == "gd":
                    descenso_gradiente(p, rng.uniform(lo, hi, n), lr=1e-3)
                elif metodo == "ga":
                    algoritmo_genetico(p, rng, lo, hi)
                elif metodo == "pso":
                    pso(p, rng, lo, hi)
                else:
                    evolucion_diferencial(p, rng, lo, hi)
                assert p.total_equivalente <= B, (nombre, n, metodo)



def test_reproducible_con_misma_semilla():
    def corrida(seed):
        p, (lo, hi) = _problema("rosenbrock", 2, 2000)
        return pso(p, np.random.default_rng(seed), lo, hi)["f"]
    assert corrida(7) == corrida(7)
    assert corrida(7) != corrida(8)


def test_los_poblacionales_resuelven_rastrigin_2d():
    for metodo in ("pso", "de"):
        p, (lo, hi) = _problema("rastrigin", 2, 10000)
        rng = np.random.default_rng(0)
        r = pso(p, rng, lo, hi) if metodo == "pso" else evolucion_diferencial(p, rng, lo, hi)
        assert r["f"] < 1e-3, metodo


def test_gd_rastrigin_cae_en_minimo_local_desde_lejos():
    p, _ = _problema("rastrigin", 2, 10000)
    r = descenso_gradiente(p, np.array([3.4, -2.6]), lr=1e-3)
    assert r["f"] > 0.9            # los minimos locales de Rastrigin tienen f >= ~0.99


def test_evals_exito_se_guarda_la_primera_vez_que_llega_a_la_meta():
    d = FUNCIONES["rastrigin"]
    p = Problema(d["f"], d["grad"], n=2, max_evals=1000, meta=1e-3)
    p.f(np.array([1.0, 1.0]))            # f = 2: aun no llega (1 evaluacion)
    assert p.evals_exito is None
    p.f(np.zeros((3, 2)))                # f = 0 en el origen: llega a las 1 + 3 = 4 evaluaciones
    assert p.evals_exito == 4
    p.f(np.zeros((5, 2)))                # no se vuelve a escribir
    assert p.evals_exito == 4