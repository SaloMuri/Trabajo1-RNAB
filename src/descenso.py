"""Descenso por gradiente con paso fijo.

    x_{k+1} = x_k - lr * grad f(x_k)

En cada iteracion se evalua el gradiente (1 grad) y f en el nuevo punto (1 f).
Para cuando: se acaba el presupuesto, el gradiente es casi cero, o el metodo diverge.
"""
import numpy as np


def descenso_gradiente(p, x0, lr=1e-3, tol_grad=1e-8, guardar_trayectoria=False, limites=None):
    x = np.asarray(x0, dtype=float).copy()
    trayectoria = [x.copy()] if guardar_trayectoria else None
    motivo = "presupuesto"
    iteraciones = 0

    with np.errstate(all="ignore"):          # con paso grande los numeros pueden desbordarse
        p.f(x)                               # f en el punto inicial
        while p.hay_presupuesto(p.costo_grad_en_f + 1):
            g = p.grad(x)
            if not np.all(np.isfinite(g)):
                motivo = "divergencia"
                break
            if np.linalg.norm(g) < tol_grad:
                motivo = "gradiente"
                break
            x = x - lr * g
            # Proyección al dominio: GD también debe respetar los límites de búsqueda.
            if limites is not None:
                lo, hi = limites
                x = np.clip(x, lo, hi)
            fx = p.f(x)
            iteraciones += 1
            if guardar_trayectoria:
                trayectoria.append(x.copy())
            if not np.isfinite(fx):
                motivo = "divergencia"
                break

    return dict(x=p.mejor_x, f=p.mejor_f, iteraciones=iteraciones, motivo=motivo,
                divergio=(motivo == "divergencia"), trayectoria=trayectoria)
