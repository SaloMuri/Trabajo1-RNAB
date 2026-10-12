"""Algoritmo genetico con codificacion real (cada individuo es un vector de n numeros).

Idea: se mantiene una poblacion; en cada generacion los mejores individuos "se reproducen"
y sus hijos reemplazan a la poblacion (menos la elite, que se conserva).

Cada generacion:
  1) Elitismo  : los `elite` mejores pasan sin cambios (y sin volver a evaluarse).
  2) Seleccion : torneo. Se sortean `k_torneo` individuos y gana el de menor f.
  3) Cruce     : BLX-alpha. Cada coordenada del hijo se sortea entre
                 [min - alpha*d, max + alpha*d], con d = |padre1 - padre2|.
                 Se aplica con probabilidad p_cruce (si no, el hijo es copia del padre).
  4) Mutacion  : cada coordenada, con probabilidad p_mut (por defecto 1/n), suma un ruido
                 normal con desviacion sigma * (ancho del dominio).
  5) Limites   : si algo se sale del dominio se recorta.

Presupuesto: la poblacion inicial cuesta `pop` evaluaciones y cada generacion cuesta
`pop - elite` (solo se evaluan los hijos).
"""
import numpy as np


def algoritmo_genetico(p, rng, lo, hi, pop=40, elite=2, k_torneo=3, alpha=0.5,
                       p_cruce=0.9, sigma=0.05, p_mut=None, guardar_trayectoria=False):
    """Corre el GA hasta agotar el presupuesto de `p`. Devuelve un dict con el mejor x y f."""
    assert p.max_evals is not None, "GA necesita un presupuesto (max_evals)"
    n = p.n
    p_mut = 1.0 / n if p_mut is None else p_mut
    n_hijos = pop - elite

    # poblacion inicial al azar dentro del dominio
    x = rng.uniform(lo, hi, (pop, n))
    fx = np.asarray(p.f(x))
    trayectoria = [x.copy()] if guardar_trayectoria else None

    generaciones = 0
    while p.hay_presupuesto(n_hijos):
        # 1) elitismo: guardar los mejores
        orden = np.argsort(fx)
        elite_x, elite_f = x[orden[:elite]], fx[orden[:elite]]

        # 2) seleccion por torneo (se piden padres de a pares)
        n_padres = n_hijos + (n_hijos % 2)
        rivales = rng.integers(0, pop, (n_padres, k_torneo))
        ganadores = rivales[np.arange(n_padres), np.argmin(fx[rivales], axis=1)]
        padres = x[ganadores]
        padre1, padre2 = padres[0::2], padres[1::2]

        # 3) cruce BLX-alpha
        minimo, maximo = np.minimum(padre1, padre2), np.maximum(padre1, padre2)
        d = maximo - minimo
        hijo1 = rng.uniform(minimo - alpha * d, maximo + alpha * d)
        hijo2 = rng.uniform(minimo - alpha * d, maximo + alpha * d)
        hay_cruce = rng.random((len(padre1), 1)) < p_cruce
        hijos = np.vstack([np.where(hay_cruce, hijo1, padre1),
                           np.where(hay_cruce, hijo2, padre2)])[:n_hijos]

        # 4) mutacion gaussiana y 5) limites del dominio
        muta = rng.random(hijos.shape) < p_mut
        hijos = hijos + muta * rng.normal(0.0, sigma * (hi - lo), hijos.shape)
        hijos = np.clip(hijos, lo, hi)

        # nueva generacion = elite + hijos (solo los hijos se evaluan)
        f_hijos = np.asarray(p.f(hijos))
        x = np.vstack([elite_x, hijos])
        fx = np.concatenate([elite_f, f_hijos])

        generaciones += 1
        if guardar_trayectoria:
            trayectoria.append(x.copy())

    return dict(x=p.mejor_x, f=p.mejor_f, iteraciones=generaciones, motivo="presupuesto",
                divergio=False, trayectoria=trayectoria)
