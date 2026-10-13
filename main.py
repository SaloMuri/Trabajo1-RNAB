"""Punto de entrada unico del Proyecto.

Uso:
    python main.py                    # Corre la Parte 1 con configs/parte1.yaml (Por defecto)
    python main.py configs/demo.yaml  # Corre la Parte 1 con un yaml específico
    python main.py parte2             # Corre la Parte 2 (TSP Combinatorio)
    python main.py sensibilidad       # Corre el análisis de la hora del vendedor
"""
import sys

# --- IMPORTS PARTE 1 (Trabajo del equipo) ---
from experimentos.gifs import generar_gifs
from experimentos.graficas import generar_graficas
from experimentos.parte1 import cargar_config, ejecutar_parte1

# --- IMPORTS PARTE 2 (Nuevos) ---
from experimentos.parte2 import correr_experimentos
from experimentos.analisis_sueldo import ejecutar_analisis

def run_parte1(ruta_config):
    cfg = cargar_config(ruta_config)
    print(f"Configuracion: {ruta_config}\n")
    ejecutar_parte1(cfg)
    print("\nGenerando gráficas estadísticas...")
    generar_graficas(cfg)
    print("\nGenerando GIFs...")
    generar_gifs(cfg)
    print(f"\nListo. Tablas y gráficas en {cfg['carpeta_resultados']}/ y GIFs en {cfg['gifs']['carpeta']}/")

def main():
    # 1. Si no pasan argumentos, respetamos la lógica original de la Parte 1 por defecto
    if len(sys.argv) == 1:
        run_parte1("configs/parte1.yaml")
        return
    
    arg = sys.argv[1]

    # Respetar la ejecución si el equipo pasa un yaml directamente
    if arg.endswith('.yaml'):
        run_parte1(arg)
        
    # 2. Nuevos comandos integrados
    elif arg == "parte1":
        ruta = sys.argv[2] if len(sys.argv) > 2 else "configs/parte1.yaml"
        run_parte1(ruta)
        
    elif arg == "parte2":
        ruta = sys.argv[2] if len(sys.argv) > 2 else "configs/parte2.yaml"
        print("Iniciando Parte 2: Optimización Combinatoria (TSP)...")
        correr_experimentos(ruta)
        
    elif arg == "sensibilidad":
        ruta = sys.argv[2] if len(sys.argv) > 2 else "configs/parte2.yaml"
        print("Iniciando Análisis de Sensibilidad (Valor vs Peajes)...")
        ejecutar_analisis(ruta)
        
    else:
        print("❌ Comando no reconocido.")
        print("Uso permitidos:")
        print("  python main.py")
        print("  python main.py configs/demo.yaml")
        print("  python main.py parte2")
        print("  python main.py sensibilidad")

if __name__ == "__main__":
    main()