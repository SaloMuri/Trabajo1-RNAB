from readline import redisplay
import pandas as pd
import unicodedata

# 1. Cargar datos
df = pd.read_csv("data/tsp/MUNICIPIOS.csv", sep=";", encoding="latin1")

# 2. Capitales objetivo
capitales = [
    "A Coruña", "Albacete", "Alacant/Alicante", "Almería", "Ávila",
    "Badajoz", "Barcelona", "Bilbao", "Burgos", "Cáceres", "Cádiz",
    "Castelló de la Plana/Castellón de la Plana", "Ciudad Real",
    "Córdoba","Cuenca", "Girona", "Granada", "Guadalajara", "Huelva",
    "Huesca", "Jaén", "León", "Lleida", "Logroño", "Lugo", "Madrid",
    "Málaga", "Murcia", "Ourense", "Oviedo", "Palencia", "Pamplona/Iruña",
    "Pontevedra", "Salamanca", "Donostia/San Sebastián", "Santander",
    "Segovia", "Sevilla", "Soria", "Tarragona", "Teruel", "Toledo",
    "Valencia", "Valladolid", "Vitoria-Gasteiz", "Zamora", "Zaragoza",
]


# 3. Función de normalización
def normalizar(texto):
  texto = unicodedata.normalize("NFKD", str(texto).strip().casefold())
  return "".join(c for c in texto if not unicodedata.combining(c))


# 4. Filtrar y mapear
df["nombre_normalizado"] = df["NOMBRE_ACTUAL"].apply(normalizar)
capitales_map = {normalizar(c): c for c in capitales}

resultado = df[df["nombre_normalizado"].isin(capitales_map)].copy()
resultado["ciudad"] = resultado["nombre_normalizado"].map(capitales_map)

# 5. Seleccionar y convertir coordenadas
coordenadas = resultado[[
    "ciudad",
    "LONGITUD_ETRS89_REGCAN95",
    "LATITUD_ETRS89_REGCAN95",
]].rename(
    columns={
        "LONGITUD_ETRS89_REGCAN95": "longitud",
        "LATITUD_ETRS89_REGCAN95": "latitud",
    }
)

for col in ["longitud", "latitud"]:
  coordenadas[col] = pd.to_numeric(
      coordenadas[col].astype(str).str.replace(",", ".", regex=False),
      errors="coerce",
  )

# 6. Eliminar duplicados y reordenar según lista original
coordenadas = coordenadas.drop_duplicates(subset="ciudad")
coordenadas["_orden"] = coordenadas["ciudad"].map(
    {c: i for i, c in enumerate(capitales)}
)
coordenadas = (
    coordenadas.sort_values("_orden")
    .drop(columns="_orden")
    .reset_index(drop=True)
)
coordenadas.insert(0, "id", range(1, len(coordenadas) + 1))

# 7. Validar
faltantes = sorted(set(capitales) - set(coordenadas["ciudad"]))
if faltantes:
  raise ValueError(f"Faltan capitales: {faltantes}")

# 8. Exportar a CSV local (descarga manualmente desde la barra lateral de Colab)
nombre_salida = "coordenadas_47_capitales.csv"
coordenadas.to_csv(nombre_salida, index=False, encoding="utf-8-sig")

print(f"✅ Archivo '{nombre_salida}' generado correctamente ({len(coordenadas)} ciudades).")
redisplay(coordenadas)