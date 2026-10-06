# Trabajo 1: Optimizacion numerica, metaheuristica y combinatoria

## Instalacion
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verificar que todo esta bien
```bash
python -m pytest -q
```

## Demo
```bash
python -m experimentos.demo_contador --funcion rosenbrock --n 2 --seed 42
```

## Semillas
Todas las corridas usan `np.random.default_rng(seed)`; la semilla se pasa por `--seed`.

## Estructura
- `src/`: funciones, contador de evaluaciones y algoritmos
- `experimentos/`: un script por experimento (un solo comando)
- `configs/`: parametros en YAML
- `data/tsp/`: datos del TSP con fuentes y fechas de consulta
- `resultados/`: tablas, figuras y GIFs (regenerables)
- `reporte/`: blog
