"""Rejilla de hiperparametros para PSO (pop, w, c1, c2) con presupuesto fijo.

Uso (desde la raiz del repositorio):

    python -m scripts.grid_search_pso                              # configs/grid_pso.yaml
    python -m scripts.grid_search_pso configs/grid_pso_demo.yaml   # version rapida (< 2 min)

Que hace, para cada (funcion, dimension):

  1. AJUSTE. Corre PSO con cada combinacion de la rejilla. Todas las combinaciones usan las
     MISMAS semillas (`corridas_ajuste` de ellas) y el MISMO presupuesto de evaluaciones, asi la
     comparacion es justa. Estas semillas son distintas de las del experimento principal.
  2. ELECCION. La combinacion ganadora es la de menor `costo_medio`: evaluaciones que hacen
     falta, en promedio, para llegar a la meta (f < f_optimo + umbral). Si una corrida no
     llega, cuenta como el presupuesto completo. Mide exito y rapidez en un solo numero.
     Desempata el menor f medio final.
  3. COMPARACION ANTES/DESPUES. Corre la configuracion por defecto (la del YAML de la parte 1)
     y la ganadora con las semillas del experimento principal (`corridas_finales` corridas),
     que NO se usaron para elegir. Asi lo reportado no esta sesgado por la eleccion.

Salidas en `carpeta_resultados`: rejilla.csv, mejores.csv, comparacion.csv, mapas de calor
(uno por funcion y dimension), curvas de convergencia (antes vs despues) y config_usada.yaml.
"""
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")                       # guardar figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from experimentos.parte1 import FUNCIONES_ORDEN, cargar_config, semillas_de_las_corridas
from src.estadisticas import resumir
from src.funciones import FUNCIONES
from src.metodos import crear_problema, ejecutar_metodo

HIPERPARAMETROS = ["pop", "w", "c1", "c2"]
# "Indice de metodo" ficticio: asi las semillas de ajuste no coinciden con las de ningun metodo.
INDICE_SEMILLA_AJUSTE = 100


# ------------------------------------------------------------------ configuracion
def cargar_config_grid(ruta):
    """Lee el YAML de la rejilla y la configuracion base de la parte 1 que referencia."""
    with open(ruta, encoding="utf-8") as archivo:
        grid = yaml.safe_load(archivo)
    cfg = cargar_config(grid["base"])
    for clave in ("presupuesto_evaluaciones", "funciones", "dimensiones"):
        if grid.get(clave) is not None:                 # el YAML de la rejilla puede cambiarlos
            cfg[clave] = grid[clave]
    for funcion in cfg["funciones"]:
        if funcion not in cfg["umbral_exito"]:
            raise ValueError(f"Falta umbral_exito para {funcion!r} en {grid['base']}")
    for h in HIPERPARAMETROS:
        if not grid["rejilla"].get(h):
            raise ValueError(f"La rejilla necesita al menos un valor para {h!r}")
    grid.setdefault("n_procesos", 1)
    return cfg, grid


def semillas_de_ajuste(cfg, funcion, n, k):
    """k semillas para el ajuste. No dependen de la combinacion: todas usan las mismas."""
    clave = [cfg["semilla"], FUNCIONES_ORDEN.index(funcion), n, INDICE_SEMILLA_AJUSTE]
    return np.random.SeedSequence(clave).spawn(k)


# ------------------------------------------------------------------ una corrida
def una_corrida(funcion, n, parametros, semilla, presupuesto, umbral):
    """Una corrida de PSO. Devuelve lo que necesita `resumir` y el historial de convergencia."""
    datos = FUNCIONES[funcion]
    meta = datos["f_opt"] + umbral
    problema = crear_problema(funcion, n, presupuesto, None, meta)
    resultado = ejecutar_metodo("pso", problema, np.random.default_rng(semilla),
                                datos["dominio"], parametros)
    return dict(f=resultado["f"], n_f=problema.n_f, n_grad=problema.n_grad,
                total=problema.total_equivalente, evals_exito=problema.evals_exito,
                divergio=resultado["divergio"], motivo=resultado["motivo"],
                iteraciones=resultado["iteraciones"], historial=problema.historial)


def resumen_de_combinacion(filas, presupuesto):
    """Numeros con los que se compara una combinacion durante el ajuste."""
    costos = [r["evals_exito"] if r["evals_exito"] is not None else presupuesto for r in filas]
    f = [r["f"] for r in filas]
    return dict(tasa_exito=float(np.mean([r["evals_exito"] is not None for r in filas])),
                costo_medio=float(np.mean(costos)),
                f_media=float(np.mean(f)),
                f_peor=float(np.max(f)))


def evaluar_combinacion(argumentos):
    """Corre todas las semillas de una combinacion (funcion de nivel superior: se puede paralelizar)."""
    funcion, n, parametros, semillas, presupuesto, umbral = argumentos
    filas = [una_corrida(funcion, n, parametros, s, presupuesto, umbral) for s in semillas]
    return resumen_de_combinacion(filas, presupuesto)


# ------------------------------------------------------------------ ajuste
def correr_rejilla(cfg, grid, funcion, n):
    """Evalua todas las combinaciones para una funcion y dimension. Devuelve un DataFrame."""
    presupuesto = cfg["presupuesto_evaluaciones"]
    umbral = cfg["umbral_exito"][funcion]
    semillas = semillas_de_ajuste(cfg, funcion, n, grid["corridas_ajuste"])
    valores = [grid["rejilla"][h] for h in HIPERPARAMETROS]
    combinaciones = [dict(zip(HIPERPARAMETROS, v)) for v in itertools.product(*valores)]

    tareas = [(funcion, n, {**cfg["pso"], **c}, semillas, presupuesto, umbral)
              for c in combinaciones]
    if grid["n_procesos"] > 1:
        with ProcessPoolExecutor(grid["n_procesos"]) as pool:
            resumenes = list(pool.map(evaluar_combinacion, tareas, chunksize=4))
    else:
        resumenes = [evaluar_combinacion(t) for t in tareas]

    filas = [dict(funcion=funcion, n=n, **c, **r) for c, r in zip(combinaciones, resumenes)]
    return pd.DataFrame(filas)


def elegir_mejor(df):
    """Fila ganadora: menor costo medio; si empatan, menor f medio."""
    return df.sort_values(["costo_medio", "f_media"], kind="stable").iloc[0]


def hiperparametros_en_el_borde(mejor, rejilla):
    """Hiperparametros cuyo mejor valor es el minimo o el maximo de la rejilla (ampliarla).

    Solo cuenta si la rejilla tiene 3 o mas valores para ese hiperparametro: con 2 valores,
    el ganador siempre esta en un borde.
    """
    en_borde = []
    for h in HIPERPARAMETROS:
        valores = sorted(rejilla[h])
        if len(valores) >= 3 and mejor[h] in (valores[0], valores[-1]):
            en_borde.append(h)
    return en_borde


# ------------------------------------------------------------------ antes vs despues
def comparar_antes_despues(cfg, grid, funcion, n, mejor):
    """Corre la configuracion por defecto y la ganadora con las semillas del experimento principal."""
    presupuesto = cfg["presupuesto_evaluaciones"]
    umbral = cfg["umbral_exito"][funcion]
    f_opt = FUNCIONES[funcion]["f_opt"]
    cfg_final = dict(cfg, corridas=grid["corridas_finales"])
    semillas = semillas_de_las_corridas(cfg_final, funcion, n, "pso")

    configuraciones = {
        "defecto": dict(cfg["pso"]),
        "ajustada": {**cfg["pso"], **{h: mejor[h].item() for h in HIPERPARAMETROS}},
    }
    filas_resumen, historiales = [], {}
    for nombre, parametros in configuraciones.items():
        corridas = [una_corrida(funcion, n, parametros, s, presupuesto, umbral) for s in semillas]
        resumen = resumir(corridas, f_opt, umbral)
        filas_resumen.append(dict(funcion=funcion, n=n, configuracion=nombre,
                                  **{h: parametros[h] for h in HIPERPARAMETROS}, **resumen))
        historiales[nombre] = [r["historial"] for r in corridas]
    return filas_resumen, historiales


# ------------------------------------------------------------------ figuras
def mapa_de_calor(df, mejor, ruta, titulo):
    """Tres cortes de la rejilla que pasan por la combinacion ganadora (se fijan los otros dos)."""
    cortes = [("pop", "w"), ("w", "c1"), ("c1", "c2")]
    vmin, vmax = df["costo_medio"].min(), df["costo_medio"].max()
    norma = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.0001))
    fig, ejes = plt.subplots(1, len(cortes), figsize=(5.4 * len(cortes), 4.6),
                             constrained_layout=True)
    for ax, (fila, col) in zip(ejes, cortes):
        corte = df
        for h in HIPERPARAMETROS:
            if h not in (fila, col):
                corte = corte[corte[h] == mejor[h]]
        tabla = corte.pivot(index=fila, columns=col, values="costo_medio").sort_index()
        tabla = tabla.reindex(sorted(tabla.columns), axis=1)
        im = ax.imshow(tabla.values, aspect="auto", cmap="viridis_r", norm=norma)
        ax.set_xticks(range(tabla.shape[1]), [f"{v:g}" for v in tabla.columns])
        ax.set_yticks(range(tabla.shape[0]), [f"{v:g}" for v in tabla.index])
        ax.set_xlabel(col)
        ax.set_ylabel(fila)
        fijos = ", ".join(f"{h}={mejor[h]:g}" for h in HIPERPARAMETROS if h not in (fila, col))
        ax.set_title(f"{fila} vs {col}  (fijos: {fijos})", fontsize=9)
        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                ax.text(j, i, f"{tabla.values[i, j]:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if norma(tabla.values[i, j]) > 0.55 else "black")
        j = list(tabla.columns).index(mejor[col])
        i = list(tabla.index).index(mejor[fila])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    fig.colorbar(im, ax=ejes, shrink=0.85, label="costo medio (evaluaciones hasta la meta)")
    fig.suptitle(titulo)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def curva_mediana(historiales, presupuesto, f_opt, puntos=400):
    """Mediana y cuartiles de (mejor f - f_opt) a lo largo de las evaluaciones."""
    inicio = max(h[0][0] for h in historiales)          # desde que todas tienen su poblacion inicial evaluada
    xs = np.linspace(inicio, presupuesto, puntos)
    matriz = np.full((len(historiales), puntos), np.nan)
    for k, historial in enumerate(historiales):
        evals = np.array([h[0] for h in historial])
        valores = np.array([h[1] for h in historial])
        pos = np.searchsorted(evals, xs, side="right") - 1      # ultima mejora hasta cada x
        ok = pos >= 0
        matriz[k, ok] = valores[pos[ok]]
    matriz = np.maximum(matriz - f_opt, 1e-16)                  # el piso permite graficar en log
    return xs, np.nanpercentile(matriz, [25, 50, 75], axis=0)


def grafica_convergencia(historiales, presupuesto, f_opt, ruta, titulo):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colores = {"defecto": "tab:blue", "ajustada": "tab:red"}
    for nombre, hist in historiales.items():
        xs, (q1, med, q3) = curva_mediana(hist, presupuesto, f_opt)
        ax.plot(xs, med, label=f"{nombre} (mediana)", color=colores[nombre])
        ax.fill_between(xs, q1, q3, alpha=0.2, color=colores[nombre])
    ax.set_yscale("log")
    ax.set_xlabel("evaluaciones de f")
    ax.set_ylabel("mejor f - f_optimo")
    ax.set_title(titulo, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ programa principal
def ejecutar_rejilla(cfg, grid):
    inicio = time.time()
    carpeta = grid["carpeta_resultados"]
    os.makedirs(carpeta, exist_ok=True)
    presupuesto = cfg["presupuesto_evaluaciones"]
    n_combos = int(np.prod([len(grid["rejilla"][h]) for h in HIPERPARAMETROS]))
    print(f"Rejilla: {n_combos} combinaciones x {grid['corridas_ajuste']} semillas de ajuste, "
          f"presupuesto {presupuesto} evaluaciones\n", flush=True)

    todas, mejores, comparacion = [], [], []
    for funcion in cfg["funciones"]:
        for n in cfg["dimensiones"]:
            caso = f"{funcion}_{n}d"
            df = correr_rejilla(cfg, grid, funcion, n)
            mejor = elegir_mejor(df)
            borde = hiperparametros_en_el_borde(mejor, grid["rejilla"])
            filas, historiales = comparar_antes_despues(cfg, grid, funcion, n, mejor)

            todas.append(df)
            mejores.append(dict(funcion=funcion, n=n, **{h: mejor[h].item() for h in HIPERPARAMETROS},
                                costo_medio=mejor["costo_medio"], tasa_exito=mejor["tasa_exito"],
                                en_el_borde=",".join(borde)))
            comparacion.extend(filas)

            mapa_de_calor(df, mejor, os.path.join(carpeta, f"mapa_calor_{caso}.png"),
                          f"{funcion}, n={n}: costo medio por combinacion "
                          f"({grid['corridas_ajuste']} semillas, presupuesto {presupuesto})")
            grafica_convergencia(historiales, presupuesto, FUNCIONES[funcion]["f_opt"],
                                 os.path.join(carpeta, f"convergencia_{caso}.png"),
                                 f"{funcion}, n={n}: por defecto vs ajustada "
                                 f"({grid['corridas_finales']} corridas con semillas distintas a las del ajuste)")

            antes, despues = filas
            print(f"{funcion:10s} n={n}  mejor: " + " ".join(f"{h}={mejor[h]:g}" for h in HIPERPARAMETROS), flush=True)
            print(f"    antes  (defecto):  exito={antes['tasa_exito']:.2f}  media_f={antes['media']:.3e}  "
                  f"evals_exito_mediana={antes['evals_exito_mediana']:.0f}")
            print(f"    despues (ajustada): exito={despues['tasa_exito']:.2f}  media_f={despues['media']:.3e}  "
                  f"evals_exito_mediana={despues['evals_exito_mediana']:.0f}")
            if borde:
                print(f"    OJO: el mejor valor de {', '.join(borde)} esta en el borde de la rejilla; conviene ampliarla.")

    pd.concat(todas, ignore_index=True).to_csv(os.path.join(carpeta, "rejilla.csv"), index=False)
    pd.DataFrame(mejores).to_csv(os.path.join(carpeta, "mejores.csv"), index=False)
    pd.DataFrame(comparacion).to_csv(os.path.join(carpeta, "comparacion.csv"), index=False)
    with open(os.path.join(carpeta, "config_usada.yaml"), "w", encoding="utf-8") as archivo:
        yaml.safe_dump(dict(base=cfg, grid=grid), archivo, sort_keys=False, allow_unicode=True)
    print(f"\nListo en {time.time() - inicio:.1f} s. Resultados en {carpeta}/")


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "configs/grid_pso.yaml"
    ejecutar_rejilla(*cargar_config_grid(ruta))
