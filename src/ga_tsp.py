"""Algoritmo Genetico (AG) para el problema del viajante (TSP).

La clase AlgoritmoGeneticoTSP busca la ruta de costo minimo que pasa una vez por cada ciudad
y vuelve a la de origen. Cada individuo es una permutacion de las ciudades y su costo se
calcula con la matriz de costos del gestor (incluyendo el regreso a la ciudad inicial).

Parametros (valores por defecto):
  tamano_poblacion = 100, num_generaciones = 300,
  prob_mutacion = 0.2 (probabilidad de mutar un hijo),
  elitismo = True (el mejor individuo pasa sin cambios a la siguiente generacion).

Resumen de lo que hace:
  ejecutar            : crea una poblacion aleatoria y, en cada generacion, evalua los costos,
                        actualiza el mejor global y genera la nueva poblacion por seleccion,
                        cruce y mutacion. Devuelve (ruta, costo).
  _crear_individuo    : permutacion aleatoria de las ciudades.
  _calcular_costo     : suma los costos de los tramos consecutivos y el de regreso al inicio.
  _seleccion_torneo   : elige k = 3 individuos al azar y gana el de menor costo.
  _cruce_ox           : Order Crossover (OX1). Copia un tramo del padre 1 y completa con las
                        ciudades faltantes en el orden del padre 2.
  _mutar_inversion    : con probabilidad prob_mutacion invierte un sub-tramo aleatorio.

Guarda el historial de mejores rutas por generacion para generar el GIF, con el mismo formato
que el ACO. Se fijan las semillas de random y numpy (Semilla = 28) para reproducir los resultados.
"""

import random
import numpy as np
Semilla=28
random.seed(Semilla)
np.random.seed(Semilla)

class AlgoritmoGeneticoTSP:
    def __init__(self, gestor, tamano_poblacion=100, num_generaciones=300, prob_mutacion=0.2, elitismo=True):
        self.gestor = gestor
        self.num_ciudades = gestor.num_ciudades

        # --- PARÁMETROS EVOLUTIVOS ---
        self.tamano_poblacion = tamano_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.elitismo = elitismo

        # Tracking para compatibilidad exacta con los reportes/GIFs
        self.mejor_ruta_global = None
        self.mejor_costo_global = float('inf')
        self.historial_mejores_rutas = []

    def ejecutar(self):
        print(f"\n🧬 Iniciando Evolución ({self.tamano_poblacion} ind, {self.num_generaciones} generaciones)...")

        # 1. Crear población inicial aleatoria
        poblacion = [self._crear_individuo() for _ in range(self.tamano_poblacion)]

        for generacion in range(self.num_generaciones):
            # 2. Evaluar el fitness (costo) de todos
            costos = [self._calcular_costo(ind) for ind in poblacion]

            # Buscar al mejor de esta generación
            mejor_idx_gen = np.argmin(costos)
            mejor_costo_gen = costos[mejor_idx_gen]

            # Actualizar récord global
            if mejor_costo_gen < self.mejor_costo_global:
                self.mejor_costo_global = mejor_costo_gen
                # Le agregamos la ciudad de inicio al final para cerrar el ciclo TSP
                self.mejor_ruta_global = poblacion[mejor_idx_gen] + [poblacion[mejor_idx_gen][0]]

            # Guardar frame para el GIF
            self.historial_mejores_rutas.append((self.mejor_ruta_global.copy(), self.mejor_costo_global))

            if generacion % 20 == 0 or generacion == self.num_generaciones - 1:
                print(f"Generación {generacion:03d} | Mejor Costo Histórico: {self.mejor_costo_global:.2f} €")

            # 3. Selección y Reproducción
            nueva_poblacion = []

            if self.elitismo:
                # El mejor clon pasa invicto
                nueva_poblacion.append(poblacion[mejor_idx_gen].copy())

            while len(nueva_poblacion) < self.tamano_poblacion:
                # Selección por torneo (elegimos 3 al azar, gana el más barato)
                padre1 = self._seleccion_torneo(poblacion, costos, k=3)
                padre2 = self._seleccion_torneo(poblacion, costos, k=3)

                # Cruzamiento
                hijo = self._cruce_ox(padre1, padre2)

                # Mutación
                hijo = self._mutar_inversion(hijo)

                nueva_poblacion.append(hijo)

            poblacion = nueva_poblacion

        print(f"\n✅ Evolución terminada. Costo mínimo encontrado: {self.mejor_costo_global:.2f} €")
        return self.mejor_ruta_global, self.mejor_costo_global

    def _crear_individuo(self):
        """Devuelve una permutación aleatoria de las 47 ciudades."""
        individuo = list(range(self.num_ciudades))
        random.shuffle(individuo)
        return individuo

    def _calcular_costo(self, individuo):
        """Calcula el costo total usando la matriz unificada del Gestor."""
        costo_total = 0
        for i in range(self.num_ciudades - 1):
            costo_total += self.gestor.matriz_costos[individuo[i]][individuo[i+1]]
        # Sumar el viaje de vuelta para cerrar la ruta
        costo_total += self.gestor.matriz_costos[individuo[-1]][individuo[0]]
        return costo_total

    def _seleccion_torneo(self, poblacion, costos, k=3):
        """Elige k individuos al azar y devuelve el que tenga el costo más bajo."""
        seleccionados_indices = random.sample(range(self.tamano_poblacion), k)
        mejor_idx = min(seleccionados_indices, key=lambda i: costos[i])
        return poblacion[mejor_idx]

    def _cruce_ox(self, padre1, padre2):
        """Order Crossover (OX1): Cruza preservando el orden relativo de las ciudades."""
        hijo = [-1] * self.num_ciudades

        # Seleccionar dos puntos de corte al azar
        p1, p2 = sorted(random.sample(range(self.num_ciudades), 2))

        # Copiar el tramo del padre 1 al hijo
        hijo[p1:p2+1] = padre1[p1:p2+1]

        # Rellenar el resto con las ciudades del padre 2 que aún no están en el hijo
        ciudades_faltantes = [ciudad for ciudad in padre2 if ciudad not in hijo]

        idx_faltantes = 0
        for i in range(self.num_ciudades):
            if hijo[i] == -1:
                hijo[i] = ciudades_faltantes[idx_faltantes]
                idx_faltantes += 1

        return hijo

    def _mutar_inversion(self, individuo):
        """Invierte el orden de un sub-tramo aleatorio. Excelente para desenredar rutas en el TSP."""
        if random.random() < self.prob_mutacion:
            p1, p2 = sorted(random.sample(range(self.num_ciudades), 2))
            individuo[p1:p2+1] = reversed(individuo[p1:p2+1])
        return individuo