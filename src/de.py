"""Evolucion diferencial DE/rand/1/bin (Storn y Price, 1997).

Idea: para cada individuo se crea un candidato combinando otros tres de la poblacion;
el candidato reemplaza al individuo solo si es mejor.

Para cada individuo i de la poblacion:
  1) Mutacion : v_i = x_r1 + F * (x_r2 - x_r3), con r1, r2, r3 distintos entre si y de i.
  2) Cruce    : el candidato u_i toma la coordenada j de v_i si rand < CR (o si j es una
                coordenada elegida al azar, para que al menos una cambie); si no, la toma de x_i.
  3) Limites  : las coordenadas de u fuera del dominio se sortean de nuevo dentro de el.
  4) Seleccion: x_i pasa a ser u_i si f(u_i) <= f(x_i).
Presupuesto: la poblacion inicial cuesta `pop` evaluaciones y cada generacion otras `pop`.
"""
import numpy as np


def _candidatos_sin_el_mismo(pop):
    """Matriz (pop, pop-1): la fila i tiene los indices 0..pop-1 sin el indice i.

    Sirve para elegir r1, r2, r3 distintos de i.
    """
    j = np.arange(pop - 1)[None, :]
    return j + (j >= np.arange(pop)[:, None])


def _tres_distintos(rng, candidatos):
    """Para cada individuo, 3 indices distintos entre si y distintos de el. Forma (pop, 3)."""
    pop = candidatos.shape[0]
    orden_al_azar = np.argsort(rng.random((pop, pop - 1)), axis=1)[:, :3]
    return np.take_along_axis(candidatos, orden_al_azar, axis=1)


def evolucion_diferencial(p, rng, lo, hi, pop=20, F=0.5, CR=0.9,
                          guardar_trayectoria=False):
    """Corre DE hasta agotar el presupuesto de `p`. Devuelve un dict con el mejor x y f."""
    assert p.max_evals is not None, "DE necesita un presupuesto (max_evals)"
    assert pop >= 4, "DE/rand/1 necesita al menos 4 individuos"
    n = p.n
    candidatos = _candidatos_sin_el_mismo(pop)

    # poblacion inicial al azar dentro del dominio
    x = rng.uniform(lo, hi, (pop, n))
    fx = np.asarray(p.f(x))
    trayectoria = [x.copy()] if guardar_trayectoria else None

    generaciones = 0
    while p.hay_presupuesto(pop):
        # 1) mutacion
        r = _tres_distintos(rng, candidatos)
        mutante = x[r[:, 0]] + F * (x[r[:, 1]] - x[r[:, 2]])

        # 2) cruce binomial (al menos una coordenada viene del mutante)
        cruza = rng.random((pop, n)) < CR
        cruza[np.arange(pop), rng.integers(0, n, pop)] = True
        prueba = np.where(cruza, mutante, x)

        # 3) las coordenadas fuera del dominio se sortean de nuevo
        fuera = (prueba < lo) | (prueba > hi)
        prueba = np.where(fuera, rng.uniform(lo, hi, (pop, n)), prueba)

        # 4) seleccion uno a uno: el candidato reemplaza al individuo si no es peor
        f_prueba = np.asarray(p.f(prueba))
        gana_prueba = f_prueba <= fx
        x[gana_prueba], fx[gana_prueba] = prueba[gana_prueba], f_prueba[gana_prueba]

        generaciones += 1
        if guardar_trayectoria:
            trayectoria.append(x.copy())

    return dict(x=p.mejor_x, f=p.mejor_f, iteraciones=generaciones, motivo="presupuesto",
                divergio=False, trayectoria=trayectoria)
