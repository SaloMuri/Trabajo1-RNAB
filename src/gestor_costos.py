import yaml
import random
import numpy as np
from src.gestor_costos import GestorCostosTSP
from src.aco import ColoniaHormigasTSP
from src.ga_tsp import AlgoritmoGeneticoTSP
from src.estadisticas import generar_informe_final
from src.animacion import generar_gif_evolucion

def correr_experimentos(config_path="configs/parte2.yaml"):
    # 1. Cargar Configuración
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    # 2. Fijar Semillas Globales
    semilla = config["semilla"]
    random.seed(semilla)
    np.random.seed(semilla)
    print(f"🌱 Entorno configurado con semilla: {semilla}")

    # 3. Inicializar Gestor
    r = config["archivos"]
    v = config["vehiculo"]
    gestor = GestorCostosTSP(
        ruta_distancias=r["distancias"],
        ruta_tiempos=r["tiempos"],
        ruta_peajes=r["peajes"],
        ruta_coord=r["coordenadas"],
        ruta_export=r["export_costos"],
        valor_hora=v["valor_hora"],
        consumo_l_100km=v["consumo_l_100km"],
        precio_litro=v["precio_litro"]
    )
    gestor.construir_matrices()

    # 4. Ejecutar Colonia de Hormigas
    conf_aco = config["hormigas"]
    aco = ColoniaHormigasTSP(
        matriz_costos=gestor.matriz_costos,
        matriz_visibilidad=gestor.matriz_visibilidad,
        ciudades=gestor.ciudades,
        num_hormigas=conf_aco["num_hormigas"],
        num_iteraciones=conf_aco["num_iteraciones"],
        alpha=conf_aco["alpha"],
        beta=conf_aco["beta"],
        rho=conf_aco["rho"],
        q=conf_aco["q"]
    )
    ruta_aco, costo_aco = aco.ejecutar()
    generar_informe_final(ruta_aco, gestor, nombre_algo="ACO")
    generar_gif_evolucion(aco, gestor, filename="resultados/gifs/evolucion_aco.gif", titulo_base="Colonia de Hormigas")

    # 5. Ejecutar Genético
    conf_ga = config["genetico"]
    ga = AlgoritmoGeneticoTSP(
        gestor=gestor,
        tamano_poblacion=conf_ga["tamano_poblacion"],
        num_generaciones=conf_ga["num_generaciones"],
        prob_mutacion=conf_ga["prob_mutacion"],
        elitismo=conf_ga["elitismo"]
    )
    ruta_ga, costo_ga = ga.ejecutar()
    generar_informe_final(ruta_ga, gestor, nombre_algo="Genético")
    generar_gif_evolucion(ga, gestor, filename="resultados/gifs/evolucion_ga.gif", titulo_base="Algoritmo Genético")

if __name__ == "__main__":
    correr_experimentos()