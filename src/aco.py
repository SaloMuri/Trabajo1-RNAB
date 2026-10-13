"""Colonia de Hormigas (ACO) para el problema del viajante (TSP).

La clase ColoniaHormigasTSP busca la ruta de costo minimo que pasa una vez por cada ciudad
y vuelve a la de origen, usando una matriz de costos y una matriz de visibilidad (1 / costo).

Parametros (valores por defecto):
  num_hormigas = 47, num_iteraciones = 100,
  alpha = 1.0 (peso de la feromona), beta = 2.0 (peso de la visibilidad),
  rho = 0.1 (evaporacion), q = 1000 (deposito de feromona).

Resumen de lo que hace:
  ejecutar                  : en cada iteracion todas las hormigas construyen una ruta, se
                              actualiza el mejor global y las feromonas. Devuelve (ruta, costo).
  _construir_ruta           : parte de una ciudad al azar, visita las demas y vuelve a la inicial.
  _elegir_siguiente_ciudad  : P_ij = tau_ij^alpha * eta_ij^beta / suma sobre no visitadas,
                              elegida por ruleta.
  _calcular_costo_ruta      : suma los costos de los tramos consecutivos.
  _actualizar_feromonas     : evaporacion tau <- (1 - rho) * tau, y cada hormiga deposita
                              q / costo_ruta en los tramos que uso (en ambos sentidos).

Guarda el historial de mejores rutas por iteracion para generar el GIF.
Se fijan las semillas de random y numpy (Semilla = 28) para reproducir los resultados.
"""

import numpy as np
import random

Semilla=28
random.seed(Semilla)
np.random.seed(Semilla)


class ColoniaHormigasTSP:
    def __init__(self, matriz_costos, matriz_visibilidad, ciudades,
                 num_hormigas=47, num_iteraciones=100,
                 alpha=1.0, beta=2.0, rho=0.1, q=1000):

        self.matriz_costos = matriz_costos
        self.matriz_visibilidad = matriz_visibilidad
        self.ciudades = ciudades
        self.num_ciudades = len(ciudades)

        # --- PARÁMETROS DEL ALGORITMO ---
        self.num_hormigas = num_hormigas # Usualmente una por ciudad
        self.num_iteraciones = num_iteraciones

        self.alpha = alpha  # Peso de la feromona (experiencia pasada)
        self.beta = beta    # Peso de la heurística/visibilidad (costo inmediato)
        self.rho = rho      # Tasa de evaporación (0.1 = 10% se evapora cada ciclo)
        self.q = q          # Constante de depósito de feromona

        # Inicializar matriz de feromonas con un valor pequeño > 0
        self.feromonas = np.ones((self.num_ciudades, self.num_ciudades)) * 0.1

        # Tracking de la mejor solución
        self.mejor_ruta_global = None
        self.mejor_costo_global = float('inf')

        # Aquí guardaremos la evolución para el GIF (Punto 4 del enunciado)
        self.historial_mejores_rutas = []

    def ejecutar(self):
        print(f"\n🐜 Iniciando Colonia de Hormigas ({self.num_hormigas} hormigas, {self.num_iteraciones} iteraciones)...")

        for iteracion in range(self.num_iteraciones):
            rutas = []
            costos = []

            # 1. Cada hormiga construye un recorrido completo
            for _ in range(self.num_hormigas):
                ruta = self._construir_ruta()
                costo = self._calcular_costo_ruta(ruta)

                rutas.append(ruta)
                costos.append(costo)

                # Actualizar el récord global si encontramos algo mejor
                if costo < self.mejor_costo_global:
                    self.mejor_costo_global = costo
                    self.mejor_ruta_global = ruta.copy()

            # 2. Actualizar feromonas (Evaporación + Depósito)
            self._actualizar_feromonas(rutas, costos)

            # Guardamos la mejor ruta actual para la animación del GIF
            self.historial_mejores_rutas.append((list(self.mejor_ruta_global), self.mejor_costo_global))

            # Log de progreso cada 10 iteraciones
            if iteracion % 10 == 0 or iteracion == self.num_iteraciones - 1:
                print(f"Iteración {iteracion:03d} | Mejor Costo Histórico: {self.mejor_costo_global:.2f} €")

        print(f"\n✅ Búsqueda terminada. Costo mínimo encontrado: {self.mejor_costo_global:.2f} €")
        return self.mejor_ruta_global, self.mejor_costo_global

    def _construir_ruta(self):
        # 1. Empezar en una ciudad aleatoria
        ciudad_actual = random.randint(0, self.num_ciudades - 1)
        ruta = [ciudad_actual]
        visitadas = set([ciudad_actual])

        # 2. Visitar las 46 ciudades restantes
        for _ in range(self.num_ciudades - 1):
            siguiente = self._elegir_siguiente_ciudad(ciudad_actual, visitadas)
            ruta.append(siguiente)
            visitadas.add(siguiente)
            ciudad_actual = siguiente

        # 3. Volver a la ciudad de origen para cerrar el ciclo (TSP)
        ruta.append(ruta[0])
        return ruta

    def _elegir_siguiente_ciudad(self, actual, visitadas):
        probabilidades = np.zeros(self.num_ciudades)

        # Calcular probabilidad solo para ciudades no visitadas
        for j in range(self.num_ciudades):
            if j not in visitadas:
                tau = self.feromonas[actual][j] ** self.alpha
                eta = self.matriz_visibilidad[actual][j] ** self.beta
                probabilidades[j] = tau * eta

        # Convertir a probabilidades reales (sumar 1)
        suma_probabilidades = np.sum(probabilidades)

        if suma_probabilidades == 0:
            # Fallback de seguridad por si hay errores de redondeo a 0
            no_visitadas = list(set(range(self.num_ciudades)) - visitadas)
            return random.choice(no_visitadas)

        probabilidades = probabilidades / suma_probabilidades

        # Elegir la siguiente ciudad basándose en la ruleta de probabilidades
        siguiente_ciudad = np.random.choice(range(self.num_ciudades), p=probabilidades)
        return siguiente_ciudad

    def _calcular_costo_ruta(self, ruta):
        costo_total = 0
        for i in range(len(ruta) - 1):
            costo_total += self.matriz_costos[ruta[i]][ruta[i+1]]
        return costo_total

    def _actualizar_feromonas(self, rutas, costos):
        # 1. Evaporación: La feromona antigua desaparece en un porcentaje (rho)
        self.feromonas = (1 - self.rho) * self.feromonas

        # 2. Depósito: Las hormigas dejan feromona inversamente proporcional al costo de su ruta
        for ruta, costo in zip(rutas, costos):
            aporte = self.q / costo  # A menor costo, mayor aporte de feromona
            for i in range(len(ruta) - 1):
                origen = ruta[i]
                destino = ruta[i+1]

                self.feromonas[origen][destino] += aporte
                # Asumimos que el rastro se deja en ambas direcciones
                self.feromonas[destino][origen] += aporte