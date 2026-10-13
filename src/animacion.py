"""GIFs de optimizacion sobre curvas de nivel (solo para n = 2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.funciones import FUNCIONES


def _cuadros_a_dibujar(total, max_cuadros):
    """Indices de las iteraciones que se dibujan (espaciados si hay demasiadas)."""
    if total <= max_cuadros:
        return np.arange(total)
    return np.unique(np.linspace(0, total - 1, max_cuadros).astype(int))


def _fondo_curvas_de_nivel(ax, funcion):
    """Dibuja el fondo: curvas de nivel de la funcion (en log) y una estrella en el minimo global."""
    info = FUNCIONES[funcion]
    lo, hi = info["dominio"]
    ejes = np.linspace(lo, hi, 300)
    X1, X2 = np.meshgrid(ejes, ejes)
    Z = info["f"](np.stack([X1, X2], axis=-1))
    ax.contourf(X1, X2, np.log10(1.0 + Z - info["f_opt"]), levels=30, cmap="viridis")
    ax.plot([info["x_opt"]], [info["x_opt"]], "r*", ms=14, label="minimo global")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def crear_gif(funcion, trayectoria, ruta, titulo, es_poblacion, max_cuadros=60, fps=10):
    """
    trayectoria : lista con un elemento por iteracion.
                  es_poblacion=False -> cada elemento es un punto (n,)      (descenso por gradiente)
                  es_poblacion=True  -> cada elemento es una poblacion (m, n) (GA, PSO, DE)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    _fondo_curvas_de_nivel(ax, funcion)

    if es_poblacion:
        puntos = ax.scatter([], [], c="white", edgecolors="k", s=25, label="individuos")
    else:
        recorrido, = ax.plot([], [], "w-", lw=1)
        punto_actual, = ax.plot([], [], "wo", mec="k", ms=7, label="iterado")
    ax.legend(loc="upper right", fontsize=8)

    cuadros = _cuadros_a_dibujar(len(trayectoria), max_cuadros)

    def dibujar(k):
        i = cuadros[k]
        if es_poblacion:
            puntos.set_offsets(trayectoria[i])
        else:
            camino = np.array(trayectoria[: i + 1])
            recorrido.set_data(camino[:, 0], camino[:, 1])
            punto_actual.set_data([camino[-1, 0]], [camino[-1, 1]])
        ax.set_title(f"{titulo}\niteracion {i}", fontsize=10)

    animacion = FuncAnimation(fig, dibujar, frames=len(cuadros), blit=False)
    animacion.save(ruta, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    
def generar_gif_evolucion(algoritmo, gestor, filename="evolucion.gif", titulo_base="Algoritmo"):
    print(f"🎨 Generando GIF de {titulo_base}... (puede tardar unos segundos)")
    df_coord_sorted = gestor.df_coord.set_index('ciudad').reindex(gestor.ciudades)
    lats = df_coord_sorted['latitud'].values
    lons = df_coord_sorted['longitud'].values
    fig, ax = plt.subplots(figsize=(10, 8))
    frames_a_mostrar = algoritmo.historial_mejores_rutas[::10]
    if algoritmo.historial_mejores_rutas[-1] not in frames_a_mostrar:
        frames_a_mostrar.append(algoritmo.historial_mejores_rutas[-1])
    def update(frame_data):
        ax.clear()
        ruta_indices, costo = frame_data
        ax.scatter(lons, lats, c='red', s=20, zorder=5)
        
        ruta_lons = [lons[i] for i in ruta_indices]
        ruta_lats = [lats[i] for i in ruta_indices]
        ax.plot(ruta_lons, ruta_lats, c='blue', linewidth=1.5, zorder=4, alpha=0.8)
        
        iteracion_idx = algoritmo.historial_mejores_rutas.index(frame_data)
        ax.set_title(f"{titulo_base} - Iteración/Gen {iteracion_idx}\nCosto: {costo:,.2f} €", fontsize=14, fontweight='bold')
        ax.axis('off')
    ani = animation.FuncAnimation(fig, update, frames=frames_a_mostrar, repeat=False)
    ani.save(filename, writer='pillow', fps=4)
    plt.close()
    print(f"✅ GIF guardado como: '{filename}'.")