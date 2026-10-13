import pandas as pd
import numpy as np

class GestorCostosTSP:
    def __init__(self, ruta_distancias, ruta_tiempos, ruta_peajes, ruta_coord,valor_hora=10.6, consumo_l_100km=3.9, precio_litro=1.84, ruta_export="matriz_costos.csv"):
        self.df_dist = pd.read_csv(ruta_distancias, index_col=0)
        self.df_tiemp = pd.read_csv(ruta_tiempos, index_col=0)
        self.df_peajes = pd.read_csv(ruta_peajes, index_col=0)
        self.df_coord = pd.read_csv(ruta_coord)
        self.ruta_export = ruta_export
        #Cantidad de ciudadades
        self.ciudades = self.df_dist.columns.tolist()
        self.num_ciudades = len(self.ciudades)
        #Parametros influyentes en el calculo de costos
        self.valor_hora = valor_hora
        self.consumo_l_100km = consumo_l_100km
        self.precio_litro = precio_litro
        # Inicializamos las matrices de costos y visibilidad
        self.matriz_costos = np.zeros((self.num_ciudades, self.num_ciudades))
        self.matriz_visibilidad = np.zeros((self.num_ciudades, self.num_ciudades))
        
    #Costo del consumo de gasolina por km, en euros/km. Se calcula como (consumo_l_100km / 100)
    def calcular_costo_por_km(self):
        return (self.consumo_l_100km / 100.0) * self.precio_litro

    
    def construir_matrices(self, verbose=True):
        costo_km = self.calcular_costo_por_km()
        distancias = self.df_dist.to_numpy()
        tiempos = self.df_tiemp.to_numpy()
        peajes = self.df_peajes.to_numpy()

        self.matriz_costos = (tiempos * self.valor_hora) + (distancias * costo_km) + peajes
        
        df_export = pd.DataFrame(self.matriz_costos, index=self.df_dist.index, columns=self.df_dist.columns)
        df_export.to_csv(self.ruta_export)

        np.fill_diagonal(self.matriz_costos, np.inf)
        self.matriz_visibilidad = 1.0 / self.matriz_costos
        
        if verbose:
            print(f"\n✅ Matriz de Costos Maestra (47x47) generada y guardada en {self.ruta_export}")