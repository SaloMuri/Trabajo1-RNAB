import matplotlib.pyplot as plt
import yaml 
import random
import numpy as np
import os
from src.gestor_costos import GestorCostosTSP
from src.aco import ColoniaHormigasTSP

def analisis_parametrico_hora(gestor, rango_horas=range(5, 45, 5)):
    """Ejecuta el modelo variando el costo de la hora para encontrar puntos de quiebre."""
    print("\n" + "="*50)
    print(" 📈 INICIANDO ANÁLISIS PARAMÉTRICO DE HORA ".center(50))
    print("="*50)

    resultados_costos = []
    resultados_peajes = []
    rutas_encontradas = []

    for v_hora in rango_horas:
        print(f"\n⏳ Probando con valor de hora: {v_hora} €/h...")

        # 1. Modificar el parámetro en el gestor y reconstruir matrices
        gestor.valor_hora = float(v_hora)
        # Desactivamos el print de construir_matrices para no saturar la consola
        gestor.construir_matrices(verbose=False) 

        # 2. Ejecutar un ACO (menos iteraciones para agilizar el análisis)
        colonia = ColoniaHormigasTSP(
            matriz_costos=gestor.matriz_costos,
            matriz_visibilidad=gestor.matriz_visibilidad,
            ciudades=gestor.ciudades,
            num_hormigas=30,
            num_iteraciones=150
        )
        ruta, costo = colonia.ejecutar(verbose=False)

        # 3. Calcular cuánto se gastó en peajes en esa ruta
        peajes_pagados = sum([gestor.df_peajes.iloc[ruta[i], ruta[i+1]] for i in range(len(ruta)-1)])

        resultados_costos.append(costo)
        resultados_peajes.append(peajes_pagados)
        rutas_encontradas.append(ruta)

        print(f"   -> Costo Total: {costo:,.2f} € | Gasto en Peajes: {peajes_pagados:,.2f} €")

    # 4. Generar el gráfico de análisis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Valor de la Hora del Vendedor (€/h)', fontsize=12)
    ax1.set_ylabel('Costo Total del Viaje (€)', color=color1, fontsize=12)
    ax1.plot(list(rango_horas), resultados_costos, marker='o', color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Gasto Total en Peajes (€)', color=color2, fontsize=12)
    ax2.plot(list(rango_horas), resultados_peajes, marker='s', color=color2, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Sensibilidad de la Ruta al Valor de la Hora del Vendedor', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Asegurar que la carpeta exista y guardar
    os.makedirs('resultados/parte2/figuras', exist_ok=True)
    ruta_guardado = 'resultados/parte2/figuras/analisis_sensibilidad_hora.png'
    plt.savefig(ruta_guardado, dpi=300)
    plt.close()

    print(f"\n✅ Análisis terminado. Gráfico guardado en '{ruta_guardado}'.")
    print("⚠️ Fíjate en la línea azul (Peajes). Si sube de golpe en algún valor, ¡ese es el punto de quiebre de la ruta!")

def ejecutar_analisis(config_path="configs/parte2.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    semilla = config["semilla"]
    random.seed(semilla)
    np.random.seed(semilla)

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
    analisis_parametrico_hora(gestor, rango_horas=[5, 10, 15, 20, 25, 30, 35, 40])

if __name__ == "__main__":
    ejecutar_analisis()