"""Optimizacion por enjambre de particulas (PSO), con peso de inercia y topologia global.

Idea: cada particula tiene una posicion x y una velocidad v. En cada paso se mueve atraida
por su mejor posicion personal y por la mejor posicion de todo el enjambre.

    v <- w*v + c1*r1*(mejor_personal - x) + c2*r2*(mejor_global - x)
    x <- x + v

  w      : inercia (cuanto conserva de la velocidad anterior)
  c1, c2 : cuanto la atrae su mejor posicion personal / la mejor global
  r1, r2 : numeros al azar en [0, 1], distintos para cada particula y coordenada

Detalles de esta implementacion:
  * La velocidad se limita a +-vmax_frac*(hi-lo).
  * Si una particula se sale del dominio se deja en la frontera y se anula esa velocidad.
  * w=0.7298 y c1=c2=1.49618 son los coeficientes de constriccion de Clerc y Kennedy.
  
Opciones (todas apagadas por defecto: con los valores por defecto el resultado es el mismo
de siempre):
  * w_fin      : si se da, la inercia baja linealmente de `w` a `w_fin` a lo largo del
                 presupuesto (explora al principio, refina al final).
  * paciencia  : si se da, para cuando el mejor valor no mejora (mas de `tol_mejora`)
                 durante `paciencia` iteraciones seguidas. El motivo queda en "estancamiento".
  * v_inicial  : "aleatoria" (sorteo en [-vmax, vmax]) o "cero" (todas parten quietas).
Presupuesto: la poblacion inicial cuesta `pop` evaluaciones y cada iteracion otras `pop`.
"""
import numpy as np


def pso(p, rng, lo, hi, pop=20, w=0.7298, c1=1.49618, c2=1.49618,
        vmax_frac=0.5, w_fin=None, paciencia=None, tol_mejora=1e-12,
        v_inicial="aleatoria", guardar_trayectoria=False):
    """Corre PSO hasta agotar el presupuesto de `p`. Devuelve un dict con el mejor x y f."""
    assert p.max_evals is not None, "PSO necesita un presupuesto (max_evals)"
    assert v_inicial in ("aleatoria", "cero"), "v_inicial debe ser 'aleatoria' o 'cero'"
    n = p.n
    vmax = vmax_frac * (hi - lo)

    # posiciones y velocidades iniciales
    x = rng.uniform(lo, hi, (pop, n))
    if v_inicial == "aleatoria":
        v = rng.uniform(-vmax, vmax, (pop, n))
    else:
        v = np.zeros((pop, n))
    fx = np.asarray(p.f(x))
    mejor_personal, f_personal = x.copy(), fx.copy()
    mejor_global = mejor_personal[np.argmin(f_personal)].copy()
    trayectoria = [x.copy()] if guardar_trayectoria else None

    # cuantas iteraciones alcanza el presupuesto (sirve para repartir la inercia decreciente)
    n_iter = max((p.max_evals - p.total_equivalente) // pop, 1)

    iteraciones = 0
    sin_mejora = 0
    motivo = "presupuesto"
    while p.hay_presupuesto(pop):
        # inercia de esta iteracion: constante, o bajando linealmente de w a w_fin
        if w_fin is None:
            w_t = w
        else:
            avance = min(iteraciones / max(n_iter - 1, 1), 1.0)
            w_t = w + (w_fin - w) * avance

        # actualizar velocidad y posicion
        r1, r2 = rng.random((2, pop, n))
        v = w_t * v + c1 * r1 * (mejor_personal - x) + c2 * r2 * (mejor_global - x)
        v = np.clip(v, -vmax, vmax)
        x = x + v

        # las que se salen del dominio quedan en el borde y pierden esa velocidad
        fuera = (x < lo) | (x > hi)
        x = np.clip(x, lo, hi)
        v[fuera] = 0.0

        # evaluar y actualizar mejores personales y el mejor global
        mejor_antes = p.mejor_f
        fx = np.asarray(p.f(x))
        mejoro = fx < f_personal
        mejor_personal[mejoro], f_personal[mejoro] = x[mejoro], fx[mejoro]
        mejor_global = mejor_personal[np.argmin(f_personal)].copy()

        iteraciones += 1
        if guardar_trayectoria:
            trayectoria.append(x.copy())

        # parada por estancamiento (opcional)
        if paciencia is not None:
            sin_mejora = 0 if p.mejor_f < mejor_antes - tol_mejora else sin_mejora + 1
            if sin_mejora >= paciencia:
                motivo = "estancamiento"
                break

    return dict(x=p.mejor_x, f=p.mejor_f, iteraciones=iteraciones, motivo=motivo,
                divergio=False, trayectoria=trayectoria)
