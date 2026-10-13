"""GIFs de optimizacion sobre curvas de nivel (solo para n = 2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import json
from pathlib import Path
from src.funciones import FUNCIONES
import matplotlib.patheffects as pe

# Contorno de España guardado en el repo (sin internet): data/ESP.geo.json
RUTA_CONTORNO = Path(__file__).resolve().parent.parent / "data" / "tsp" / "ESP.geo.json"


def _cargar_contorno_espana(ruta=RUTA_CONTORNO):
    """Lee el contorno de España del archivo local. Si falla, devuelve [] (se dibuja sin fondo)."""
    try:
        with open(ruta, encoding="utf-8") as f:
            geom = json.load(f)["features"][0]["geometry"]
        poligonos = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        return [np.array(p[0]) for p in poligonos]
    except Exception as e:
        print(f"⚠️ No se pudo cargar el contorno de España ({e}). Se dibuja sin fondo.")
        return []

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

# Combinatoria (Hormigas y Evolutivo)

def generar_gif_evolucion(algoritmo, gestor, filename="evolucion.gif", titulo_base="Algoritmo"):
    """Crea una animación GIF mostrando la evolución de la ruta uniendo las capitales."""
    print(f"🎨 Generando GIF de {titulo_base}... (tardará unos segundos)")

    # 1. Preparar coordenadas
    df_coord_sorted = gestor.df_coord.set_index('ciudad').reindex(gestor.ciudades)
    lats = df_coord_sorted['latitud'].values
    lons = df_coord_sorted['longitud'].values

    # 2. Configurar figura
    fig, ax = plt.subplots(figsize=(10, 8))
    contorno = _cargar_contorno_espana()

    # Nombres de las ciudades (sin la parte despues de "/") e inicio fijo para todos los cuadros
    nombres = [n.split("/")[0] for n in df_coord_sorted.index]
    inicio_fijo = algoritmo.historial_mejores_rutas[-1][0][0]

    def rotar(ruta):
       """Reordena el ciclo para que empiece siempre en inicio_fijo (no cambia el recorrido)."""
       ciclo = list(ruta[:-1])
       k = ciclo.index(inicio_fijo)
       ciclo = ciclo[k:] + ciclo[:k]
       return ciclo + [ciclo[0]]

    # Tomamos un frame cada 10 iteraciones/generaciones para que el GIF no sea eterno
    frames_a_mostrar = algoritmo.historial_mejores_rutas[::10]
    
    # Aseguramos que el frame final siempre esté
    if algoritmo.historial_mejores_rutas[-1] not in frames_a_mostrar:
        frames_a_mostrar.append(algoritmo.historial_mejores_rutas[-1])

    def update(frame_data):
        ax.clear()
        ruta_indices, costo = frame_data
        ruta_indices = rotar(ruta_indices)

        # Fondo claro
        fig.patch.set_facecolor('#f0f4f8')
        ax.set_facecolor('#f0f4f8')

        # Límites geográficos de España peninsular para enmarcar bien la gráfica
        ax.set_xlim([-10, 4.5])
        ax.set_ylim([35.5, 44.5])

        for pol in contorno:
           ax.fill(pol[:, 0], pol[:, 1], color="white", ec="#9aa5b1", lw=1, zorder=0)

        # Dibujar las 47 capitales como puntos base (color gris claro)
        ax.scatter(lons, lats, c='lightgray', s=30, zorder=2)

        # Nombres de las ciudades
        for x, y, nom in zip(lons, lats, nombres):
            ax.annotate(nom, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7,
                     zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # Dibujar las capitales de la ruta óptima de este frame (color rojo oscuro)
        ruta_lons = [lons[i] for i in ruta_indices]
        ruta_lats = [lats[i] for i in ruta_indices]
        ax.scatter(ruta_lons, ruta_lats, c='darkred', s=40, zorder=5)

        # Conectar los puntos de la ruta en azul (simula el recorrido)
        ax.plot(ruta_lons, ruta_lats, c='#1f77b4', linewidth=2, zorder=4, alpha=0.8)

        # Marcar la ciudad de inicio/fin en verde y un poco más grande
        ax.scatter(ruta_lons[0], ruta_lats[0], c='green', s=100, zorder=6, marker='*')

        # Títulos y formato
        iteracion_idx = algoritmo.historial_mejores_rutas.index(frame_data)
        ax.set_title(f"{titulo_base} - Evolución (Iter/Gen: {iteracion_idx})\nCosto de esta ruta: {costo:,.2f} €",
                     fontsize=14, fontweight='bold', color='#333333')
        
        # Ocultar los ejes para que parezca un mapa
        ax.axis('off')

    # Generar y guardar la animación
    ani = animation.FuncAnimation(fig, update, frames=frames_a_mostrar, repeat=False)
    ani.save(filename, writer='pillow', fps=4)
    plt.close()

    print(f"✅ GIF guardado como: '{filename}'.")