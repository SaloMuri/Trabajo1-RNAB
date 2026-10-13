
import contextlib
import io
import itertools
import random
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))      # la raíz del proyecto, para poder hacer "from src..."

from src.aco import ColoniaHormigasTSP
from src.gestor_costos import GestorCostosTSP

# ------------------------------------------------------------------
# Rutas del proyecto
# ------------------------------------------------------------------
DATA = RAIZ / "data"
RESULTADOS = RAIZ / "results" / "grid_search"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Configuracion de la rejilla
# ------------------------------------------------------------------
GRID = {
    "num_hormigas": [20, 47, 80],
    "alpha":        [0.5, 1.0, 2.0],
    "beta":         [1.0, 2.0, 3.0, 5.0],
    "rho":          [0.05, 0.1, 0.3, 0.5],
}
NUM_ITERACIONES = 50          # fijo para todas las configuraciones
SEMILLAS = [11, 22, 33]       # replicas por configuracion (el ACO es estocastico)
N_JOBS = -1                   # -1 = todos los nucleos; 1 = secuencial
VALOR_HORA = 12.0             # EUR/h usado en la matriz de costos


# ------------------------------------------------------------------
# Version vectorizada del ACO (mismo comportamiento, mucho mas rapida).
# Solo sobreescribe la eleccion de la siguiente ciudad.
# ------------------------------------------------------------------
class ColoniaHormigasRapida(ColoniaHormigasTSP):
    def _elegir_siguiente_ciudad(self, actual, visitadas):
        n = self.num_ciudades
        libres = np.ones(n, dtype=bool)
        libres[list(visitadas)] = False

        tau = self.feromonas[actual] ** self.alpha
        eta = self.matriz_visibilidad[actual] ** self.beta
        prob = tau * eta * libres

        suma = prob.sum()
        if suma == 0 or not np.isfinite(suma):
            return random.choice(np.flatnonzero(libres).tolist())
        return np.random.choice(n, p=prob / suma)


# ------------------------------------------------------------------
# Una ejecucion individual (configuracion + semilla)
# ------------------------------------------------------------------
def correr_una(cfg, semilla, matriz_costos, matriz_vis, ciudades, iteraciones):
    random.seed(semilla)
    np.random.seed(semilla)
    colonia = ColoniaHormigasRapida(
        matriz_costos=matriz_costos,
        matriz_visibilidad=matriz_vis,
        ciudades=ciudades,
        num_hormigas=cfg["num_hormigas"],
        num_iteraciones=iteraciones,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        rho=cfg["rho"],
    )
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):   # silenciar los prints del ACO
        _, costo = colonia.ejecutar()
    return {**cfg, "semilla": semilla, "costo": costo, "tiempo_s": time.time() - t0}


# ------------------------------------------------------------------
# Busqueda por rejilla
# ------------------------------------------------------------------
def busqueda_rejilla(gestor, grid, semillas, iteraciones, n_jobs=-1):
    nombres = list(grid.keys())
    configs = [dict(zip(nombres, v)) for v in itertools.product(*grid.values())]
    tareas = [(c, s) for c in configs for s in semillas]

    # Estimacion de tiempo con una corrida de la configuracion mas pesada (mas hormigas)
    cfg_pesada = max(configs, key=lambda c: c["num_hormigas"])
    prueba = correr_una(cfg_pesada, semillas[0], gestor.matriz_costos,
                        gestor.matriz_visibilidad, gestor.ciudades, iteraciones)
    prom_hormigas = np.mean([c["num_hormigas"] for c in configs])
    est_min = prueba["tiempo_s"] * (prom_hormigas / cfg_pesada["num_hormigas"]) * len(tareas) / 60
    print(f"Configuraciones: {len(configs)} | Corridas totales: {len(tareas)}")
    print(f"Estimado secuencial ~ {est_min:.1f} min (dividir entre el numero de nucleos si N_JOBS=-1)")

    resultados = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(correr_una)(c, s, gestor.matriz_costos, gestor.matriz_visibilidad,
                            gestor.ciudades, iteraciones)
        for c, s in tareas
    )
    return pd.DataFrame(resultados)


# ------------------------------------------------------------------
# Graficas
# ------------------------------------------------------------------
def grafica_efecto_marginal(df_runs, cols_param):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, p in zip(axes, cols_param):
        g = df_runs.groupby(p)["costo"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g[p].astype(str), g["mean"], yerr=g["std"], marker="o", capsize=4, linewidth=2)
        ax.set_title(p, fontweight="bold")
        ax.set_xlabel(p)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Costo medio (EUR)")
    fig.suptitle("Efecto marginal de cada parametro sobre el costo (promedio sobre el resto)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "grid_efecto_marginal.png", dpi=200)
    plt.close(fig)


def grafica_heatmap_alpha_beta(df_runs):
    pivot = df_runs.pivot_table(index="alpha", columns="beta", values="costo", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center",
                    color="white", fontsize=9)
    plt.colorbar(im, ax=ax, label="Costo medio (EUR)")
    ax.set_title("Costo medio segun alpha y beta", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "grid_heatmap_alpha_beta.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------------
def main():
    # Gestor de costos con los datos del proyecto
    gestor = GestorCostosTSP(
        str(DATA / "matriz_distancias_km.csv"),
        str(DATA / "matriz_tiempos_horas.csv"),
        str(DATA / "matriz_peajes_euros.csv"),
        str(DATA / "coordenadas_47_capitales.csv"),
    )
    gestor.valor_hora = VALOR_HORA
    with contextlib.redirect_stdout(io.StringIO()):
        gestor.construir_matrices()

    # 1. Busqueda por rejilla
    df_runs = busqueda_rejilla(gestor, GRID, SEMILLAS, NUM_ITERACIONES, N_JOBS)
    df_runs.to_csv(RESULTADOS / "grid_search_aco_corridas.csv", index=False)

    # 2. Resumen por configuracion (promedio sobre las semillas)
    cols_param = list(GRID.keys())
    df_resumen = (
        df_runs.groupby(cols_param)
        .agg(costo_medio=("costo", "mean"),
             costo_std=("costo", "std"),
             costo_min=("costo", "min"),
             tiempo_medio_s=("tiempo_s", "mean"))
        .reset_index()
        .sort_values("costo_medio")
        .reset_index(drop=True)
    )
    df_resumen.to_csv(RESULTADOS / "grid_search_aco_resumen.csv", index=False)

    print("\nTop 10 configuraciones (por costo medio):")
    print(df_resumen.head(10).round(3).to_string())

    mejor = df_resumen.iloc[0]
    print("\nMejor configuracion:")
    print(f"   num_hormigas = {int(mejor['num_hormigas'])}")
    print(f"   alpha        = {mejor['alpha']}")
    print(f"   beta         = {mejor['beta']}")
    print(f"   rho          = {mejor['rho']}")
    print(f"   costo medio  = {mejor['costo_medio']:.2f} EUR (+-{mejor['costo_std']:.2f}) "
          f"| mejor corrida = {mejor['costo_min']:.2f} EUR")

    # 3. Graficas
    grafica_efecto_marginal(df_runs, cols_param)
    grafica_heatmap_alpha_beta(df_runs)

    # 4. Validacion: mejor configuracion con mas iteraciones y semillas nuevas
    cfg_final = {k: (int(mejor[k]) if k == "num_hormigas" else float(mejor[k])) for k in cols_param}
    res_val = [correr_una(cfg_final, s, gestor.matriz_costos, gestor.matriz_visibilidad,
                          gestor.ciudades, 200) for s in [101, 202, 303, 404, 505]]
    costos_val = [r["costo"] for r in res_val]
    print(f"\nValidacion (200 iteraciones, 5 semillas nuevas): "
          f"media = {np.mean(costos_val):.2f} EUR | mejor = {np.min(costos_val):.2f} EUR "
          f"| std = {np.std(costos_val):.2f}")
    print(f"\nResultados guardados en: {RESULTADOS}")


if __name__ == "__main__":
    main()