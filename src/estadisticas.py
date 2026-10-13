"""Resumen estadistico de las corridas independientes."""
import numpy as np


def resumir(filas, f_opt, umbral):
    """filas: lista de dicts con f, n_f, n_grad, total, divergio. Exito: f - f_opt < umbral."""
    f = np.array([r["f"] for r in filas], dtype=float)
    # evaluaciones hasta el exito: solo de las corridas que llegaron (las demas no tienen valor)
    ev_exito = [r["evals_exito"] for r in filas if r["evals_exito"] is not None]
    return dict(
        corridas=len(filas),
        media=float(np.mean(f)),
        desv=float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
        mejor=float(np.min(f)),
        peor=float(np.max(f)),
        tasa_exito=float(np.mean((f - f_opt) < umbral)),
        n_f_media=float(np.mean([r["n_f"] for r in filas])),
        n_grad_media=float(np.mean([r["n_grad"] for r in filas])),
        total_eq_media=float(np.mean([r["total"] for r in filas])),
        evals_exito_mediana=float(np.median(ev_exito)) if ev_exito else np.nan,
        divergencias=int(sum(bool(r["divergio"]) for r in filas)),
    )

#Combinatoria (Hormigas y Evolutivo)

def generar_informe_final(ruta_indices, gestor, nombre_algo="Algoritmo"):
    """Genera un reporte detallado de los costos y decisiones de la ruta óptima."""
    distancia_total = 0.0
    tiempo_total = 0.0
    costo_peajes_pagados = 0.0

    tramos_peaje_pagados = []
    aristas_ruta = set()

    for i in range(len(ruta_indices) - 1):
        idx_origen = ruta_indices[i]
        idx_destino = ruta_indices[i+1]

        nom_origen = gestor.ciudades[idx_origen]
        nom_destino = gestor.ciudades[idx_destino]

        dist = gestor.df_dist.iloc[idx_origen, idx_destino]
        tiemp = gestor.df_tiemp.iloc[idx_origen, idx_destino]
        peaje = gestor.df_peajes.iloc[idx_origen, idx_destino]

        distancia_total += dist
        tiempo_total += tiemp
        costo_peajes_pagados += peaje

        aristas_ruta.add((idx_origen, idx_destino))
        aristas_ruta.add((idx_destino, idx_origen))

        if peaje > 0:
            tramos_peaje_pagados.append((nom_origen, nom_destino, peaje))

    tramos_peaje_evitados = []
    for i in range(gestor.num_ciudades):
        for j in range(i + 1, gestor.num_ciudades):
            peaje_existente = gestor.df_peajes.iloc[i, j]
            if peaje_existente > 0 and (i, j) not in aristas_ruta:
                nom_i = gestor.ciudades[i]
                nom_j = gestor.ciudades[j]
                tramos_peaje_evitados.append((nom_i, nom_j, peaje_existente))

    costo_combustible = distancia_total * gestor.calcular_costo_por_km()
    costo_sueldo = tiempo_total * gestor.valor_hora
    costo_total = costo_combustible + costo_sueldo + costo_peajes_pagados

    print("\n" + "="*50)
    print(f" 📊 INFORME FINAL DE LA RUTA ({nombre_algo}) ".center(50, "="))
    print("="*50)
    print(f"🔹 Distancia total a recorrer: {distancia_total:,.2f} km")
    print(f"🔹 Tiempo total al volante:    {tiempo_total:,.2f} horas")
    print("-" * 50)
    print(f"💶 Costo de Combustible:       {costo_combustible:,.2f} €")
    print(f"💶 Costo de Horas (Sueldo):    {costo_sueldo:,.2f} €")
    print(f"💶 Costo de Peajes Pagados:    {costo_peajes_pagados:,.2f} €")
    print(f"🔥 COSTO TOTAL DEL VIAJE:      {costo_total:,.2f} €")
    print("="*50)

    print("\n🛣️  PEAJES PAGADOS (La hormiga decidió que ahorraban dinero):")
    if tramos_peaje_pagados:
        for orig, dest, precio in tramos_peaje_pagados:
            print(f"   - {orig} <-> {dest}: {precio:.2f} €")
    else:
        print("   - ¡Ninguno! La hormiga esquivó todos los peajes.")

    print("\n🚧 PEAJES EVITADOS (La hormiga prefirió dar un rodeo o ir por nacional):")
    if tramos_peaje_evitados:
        for orig, dest, precio in tramos_peaje_evitados:
            print(f"   - {orig} <-> {dest}: {precio:.2f} €")
    else:
        print("   - Ninguno evitado.")
    print("="*50 + "\n")