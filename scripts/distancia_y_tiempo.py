import pandas as pd
import requests

# 1. Cargar el CSV con las 47 capitales peninsulares
archivo_csv = "coordenadas_47_capitales.csv"
df_ciudades = pd.read_csv(archivo_csv)

# IMPORTANTE: OSRM requiere el formato "longitud,latitud" (X, Y)
coordenadas_osrm = ";".join(
    f"{lon},{lat}" for lon, lat in zip(df_ciudades["longitud"], df_ciudades["latitud"])
)

# 2. Construir la URL de la consulta a OSRM (Table Service) 
url_osrm = f"http://router.project-osrm.org/table/v1/driving/{coordenadas_osrm}?annotations=distance,duration"

print("Consultando OSRM para obtener la matriz de ruteo de 47x47...")
respuesta = requests.get(url_osrm)

if respuesta.status_code == 200:
    datos = respuesta.json()
    # OSRM entrega distancias en metros y tiempos en segundos
    # Convertimos distancias a kilómetros (m -> km)
    matriz_distancias_km = [[d / 1000.0 for d in fila] for fila in datos["distances"]]

    # Convertimos tiempos a horas (s -> h)
    matriz_tiempos_horas = [[t / 3600.0 for t in fila] for fila in datos["durations"]]

    # 3. Formatear DataFrames con los nombres de las ciudades
    nombres_ciudades = df_ciudades["ciudad"].tolist()

    df_distancias = pd.DataFrame(matriz_distancias_km, index=nombres_ciudades, columns=nombres_ciudades)
    df_tiempos = pd.DataFrame(matriz_tiempos_horas, index=nombres_ciudades, columns=nombres_ciudades)

    # 4. Exportar a CSV local
    df_distancias.to_csv("matriz_distancias_km.csv", float_format="%.2f")
    df_tiempos.to_csv("matriz_tiempos_horas.csv", float_format="%.2f")

    print("✅ Archivos 'matriz_distancias_km.csv' y 'matriz_tiempos_horas.csv' generados con éxito.")
else:
    print(f"❌ Error al consultar OSRM. Código de respuesta: {respuesta.status_code}")