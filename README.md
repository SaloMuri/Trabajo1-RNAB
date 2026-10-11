# Trabajo 1 - Parte 1: Optimizacion numerica

Compara descenso por gradiente (GD) con algoritmo genetico (GA), PSO y evolucion diferencial (DE)
sobre Rosenbrock y Rastrigin, en 2 y 3 dimensiones.

## Instalacion
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar (un solo comando)
```bash
python main.py                      # experimento completo (30 corridas) + GIFs, ~1 min
python main.py configs/demo.yaml    # version rapida (5 corridas), ~20 s: usar en la sustentacion
python -m pytest -q                 # tests
```

## Cambiar parametros
Todo esta en `configs/parte1.yaml` (semilla, numero de corridas, funciones, dimensiones, presupuesto,
costo del gradiente, umbral de exito, hiperparametros de cada metodo y lista de GIFs).
No hay que editar codigo. `configs/demo.yaml` tiene la misma estructura con valores mas pequenos.

## Como se cuentan las evaluaciones
* `Problema` (`src/contador.py`) cuenta las evaluaciones de f y de grad f por separado
  (`n_f`, `n_grad`); una poblacion de m individuos cuenta m evaluaciones de f.
* Un gradiente equivale a `costo_gradiente` evaluaciones de f. Por defecto 2n (lo que costaria
  aproximarlo con diferencias centrales en dimension n).
* Todos los metodos tienen el mismo presupuesto: `n_f + costo_gradiente * n_grad <= presupuesto_evaluaciones`.
* Exito de una corrida: `f_mejor - f_optimo < umbral_exito` (umbral distinto por funcion).

## Semillas
Todas las corridas usan `np.random.default_rng`. Las semillas de las corridas salen de
`SeedSequence([semilla, indice_funcion, n, indice_metodo]).spawn(corridas)`, con `semilla` del YAML (42).
Cada GIF usa su propia semilla (`gifs.semilla`).

## Salidas
* `resultados/parte1/`: `corridas.csv` (una fila por corrida), `resumen.csv` y `resumen.md`
  (media, desviacion, mejor, peor, tasa de exito, `n_f`, `n_grad`, total equivalente y
  `evals_exito_mediana`) y `config_usada.yaml`.
  `evals_exito_mediana` es la mediana de evaluaciones equivalentes hasta alcanzar el umbral de
  exito, calculada solo entre las corridas exitosas (vacia si ninguna llego).
* `resultados/gifs/`: GIFs de GD y DE sobre curvas de nivel (en 2D).


## Estructura
- `main.py`: punto de entrada
- `configs/`: parametros en YAML
- `src/`: funciones de prueba, contador, `descenso.py`, `ga.py`, `pso.py`, `de.py`, `metodos.py`, `estadisticas.py`, `animacion.py`
- `experimentos/`: `parte1.py` (experimento) y `gifs.py`
- `tests/`: gradientes, conteo de evaluaciones, presupuesto, reproducibilidad y experimento pequeno
- `resultados/`: se regenera con el codigo
