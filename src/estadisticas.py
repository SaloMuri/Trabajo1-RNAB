"""Resumen estadistico de las corridas independientes."""
import numpy as np


def resumir(filas, f_opt, umbral):
    """filas: lista de dicts con f, n_f, n_grad, total, divergio. Exito: f - f_opt < umbral."""
    f = np.array([r["f"] for r in filas], dtype=float)
    return dict(
        corridas=len(filas),
        media=float(np.mean(f)),
        desv=float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
        mejor=float(np.min(f)),
        peor=float(np.max(f)),
        tasa_exito=float(np.mean((f - f_opt) < umbral)),
        n_f_media=float(np.mean([r["n_f"] for r in filas])),
        n_grad_media=float(np.mean([r["n_grad"] for r in filas])),
        total_eq_media=float(np.mean([r["total"] for r in filas])),
        divergencias=int(sum(bool(r["divergio"]) for r in filas)),
    )
