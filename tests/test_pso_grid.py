"""Pruebas de PSO (opciones nuevas) y de la rejilla de hiperparametros."""
import numpy as np
import pandas as pd
import pytest

from experimentos.parte1 import cargar_config, semillas_de_las_corridas
from scripts.grid_search_pso import (elegir_mejor, hiperparametros_en_el_borde,
                                     semillas_de_ajuste, una_corrida)
from src.funciones import FUNCIONES
from src.metodos import crear_problema, ejecutar_metodo
from src.pso import pso


def correr(parametros, semilla=1, presupuesto=4000, funcion="rosenbrock", n=2):
    problema = crear_problema(funcion, n, presupuesto)
    lo, hi = FUNCIONES[funcion]["dominio"]
    resultado = pso(problema, np.random.default_rng(semilla), lo, hi, **parametros)
    return problema, resultado


def test_pso_gasta_exactamente_el_presupuesto():
    problema, resultado = correr(dict(pop=20))
    assert problema.n_f == 4000
    assert resultado["motivo"] == "presupuesto"


def test_misma_semilla_mismo_resultado():
    _, a = correr(dict(pop=20), semilla=7)
    _, b = correr(dict(pop=20), semilla=7)
    assert a["f"] == b["f"]
    np.testing.assert_array_equal(a["x"], b["x"])


def test_inercia_decreciente_cambia_la_busqueda_y_respeta_el_presupuesto():
    _, fija = correr(dict(pop=20))
    problema, decreciente = correr(dict(pop=20, w_fin=0.3))
    assert problema.n_f == 4000
    assert decreciente["f"] != fija["f"]


def test_estancamiento_para_antes_del_presupuesto():
    problema, resultado = correr(dict(pop=20, paciencia=30), presupuesto=40000)
    assert resultado["motivo"] == "estancamiento"
    assert problema.n_f < 40000


def test_velocidad_inicial_cero_es_valida():
    problema, resultado = correr(dict(pop=20, v_inicial="cero"))
    assert problema.n_f == 4000 and np.isfinite(resultado["f"])


def test_velocidad_inicial_invalida_falla():
    with pytest.raises(AssertionError):
        correr(dict(pop=20, v_inicial="otra"))


def test_historial_mejora_siempre_y_las_evaluaciones_crecen():
    problema, _ = correr(dict(pop=20))
    evals = [h[0] for h in problema.historial]
    valores = [h[1] for h in problema.historial]
    assert evals == sorted(evals) and evals[0] == 20
    assert all(b < a for a, b in zip(valores, valores[1:]))
    assert valores[-1] == problema.mejor_f


def test_ejecutar_metodo_ignora_parametros_nulos():
    """En el YAML se puede dejar `w_fin: null` sin que cambie nada."""
    problema = crear_problema("rastrigin", 2, 1000)
    lo, hi = FUNCIONES["rastrigin"]["dominio"]
    ejecutar_metodo("pso", problema, np.random.default_rng(0), (lo, hi),
                    dict(pop=10, w_fin=None, paciencia=None))
    assert problema.n_f == 1000


def test_semillas_de_ajuste_son_reproducibles_y_distintas_de_las_finales():
    cfg = cargar_config("configs/parte1.yaml")
    a = [s.generate_state(2).tolist() for s in semillas_de_ajuste(cfg, "rosenbrock", 2, 10)]
    b = [s.generate_state(2).tolist() for s in semillas_de_ajuste(cfg, "rosenbrock", 2, 10)]
    assert a == b
    finales = [s.generate_state(2).tolist()
               for metodo in ("gd", "ga", "pso", "de")
               for s in semillas_de_las_corridas(cfg, "rosenbrock", 2, metodo)]
    assert not any(estado in finales for estado in a)


def test_una_corrida_usa_el_mismo_presupuesto_para_cualquier_combinacion():
    semilla = np.random.SeedSequence(5)
    for pop in (10, 40):
        r = una_corrida("rastrigin", 2, dict(pop=pop, w=0.7, c1=1.5, c2=1.5), semilla, 2000, 1e-3)
        assert r["n_f"] == 2000


def test_elegir_mejor_prefiere_menor_costo_y_desempata_por_f():
    df = pd.DataFrame(dict(costo_medio=[500.0, 300.0, 300.0], f_media=[0.0, 1e-3, 1e-6]))
    assert elegir_mejor(df).name == 2


def test_borde_solo_cuenta_con_tres_o_mas_valores():
    rejilla = dict(pop=[10, 20], w=[0.4, 0.7, 0.9], c1=[1.0], c2=[0.5, 1.0, 2.0])
    mejor = dict(pop=10, w=0.7, c1=1.0, c2=2.0)
    assert hiperparametros_en_el_borde(mejor, rejilla) == ["c2"]
