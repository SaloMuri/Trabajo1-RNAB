Proyecto de Algoritmos de Optimización y Metaheurísticas

Este repositorio contiene la implementación y el análisis de diversos algoritmos de optimización (exactos y metaheurísticos) desarrollados en Python. El proyecto está estructurado para resolver tanto problemas de optimización continua como problemas de optimización combinatoria.

📺 Aportes Personales y Explicación (Videos de YouTube)

A continuación, se presentan los enlaces a los videos explicativos detallando los aportes personales realizados en este proyecto:

🎥 Video Aporte Parte 1 (Optimización Continua):
   Video Salome Murillo Gaviria: https://youtu.be/CFZ9XmSOUns
   Video Oky Ruiz de la Rosa: https://youtu.be/cnhzBbiD9UE?si=g88Om7LbiK1w2WYi 
🎥 Video Aporte Parte 2 (Problema del Viajante - TSP): 
    Video Nicolas David Zambrano Murcia: https://youtu.be/QMLBV_nOB4c 
    Video Andres Perez Cano: https://www.youtube.com/watch?v=bGIumkBk4j8 

📋 Descripción del Proyecto

El proyecto está dividido en dos partes principales:

Parte 1: Optimización Continua en Funciones de Prueba

Implementación de algoritmos como Particle Swarm Optimization (PSO), Differential Evolution (DE), Algoritmos Genéticos (GA) y Descenso de Gradiente (GD).

Evaluación sobre funciones benchmark clásicas como Rastrigin y Rosenbrock (en 2D y 3D).

Búsqueda de hiperparámetros (Grid Search) y análisis de convergencia.

Parte 2: Problema del Viajante (TSP) en España

Resolución del TSP aplicado a 47 capitales/municipios de España.

Uso de algoritmos como Ant Colony Optimization (ACO) y Algoritmos Genéticos (GA) adaptados para TSP.

Optimización basada en diferentes matrices de costos (distancias en km, tiempos en horas, peajes en euros).

📁 Estructura del Repositorio

├── configs/              # Archivos YAML de configuración para los experimentos (demo, pso, parte1, parte2)
├── data/                 # Conjuntos de datos (Ej: coordenadas de España, matrices de distancia/tiempo/costo para TSP)
├── experimentos/         # Scripts para la ejecución de pruebas específicas, generación de gráficas y GIFs
├── reporte/              # Reportes autogenerados (Ej: informe_final.html)
├── resultados/           # Outputs de las ejecuciones:
│   ├── gifs/             # Animaciones de la evolución de los algoritmos (aco, ga, rastrigin, rosenbrock)
│   ├── parte1/           # CSVs y figuras de convergencia, mapas de calor y Grid Search
│   └── parte2/           # Análisis de sensibilidad, cálculo de rutas y resultados TSP
├── scripts/              # Utilidades para generación de coordenadas y grid search (ACO/PSO)
├── src/                  # Código fuente (Módulos principales del proyecto)
│   ├── aco.py            # Optimización por Colonia de Hormigas
│   ├── pso.py            # Optimización por Enjambre de Partículas
│   ├── ga.py & ga_tsp.py # Algoritmos Genéticos (Continuo y Combinatorio)
│   ├── de.py             # Evolución Diferencial
│   ├── descenso.py       # Descenso de gradiente
│   ├── funciones.py      # Funciones objetivo (Rastrigin, Rosenbrock)
│   ├── animacion.py      # Lógica para la creación de GIFs
│   └── gestor_costos.py  # Manejo de las matrices para el TSP
├── tests/                # Pruebas unitarias para asegurar el correcto funcionamiento (pytest)
├── main.py               # Script principal de ejecución
├── pytest.ini            # Configuración de pytest
└── requirements.txt      # Dependencias del proyecto


⚙️ Instalación y Requisitos

Para ejecutar este proyecto, se recomienda crear un entorno virtual (venv o conda). Asegúrate de tener Python 3.8+ instalado.

Clona este repositorio:

git clone <url-del-repositorio>
cd <nombre-del-repo>


Instala las dependencias necesarias:

pip install -r requirements.txt


🚀 Uso

Para ejecutar el programa principal:

python main.py


Si deseas correr un experimento específico (por ejemplo, los de la Parte 1 o Parte 2), puedes ejecutar los scripts dentro de la carpeta experimentos/ o scripts/:

python experimentos/parte1.py
python scripts/grid_search_aco.py


La configuración de los parámetros para cada corrida se puede modificar fácilmente en los archivos .yaml dentro de la carpeta configs/.

🧪 Pruebas (Testing)

El proyecto utiliza pytest para las pruebas unitarias. Para ejecutar toda la batería de pruebas y asegurar que los algoritmos divergen/convergen correctamente y las funciones matemáticas operan como se espera, ejecuta:

pytest


📊 Resultados y Visualizaciones

Todos los resultados se guardan automáticamente en la carpeta resultados/. Encontrarás:

Animaciones .gif que muestran cómo convergen las partículas o las hormigas iteración tras iteración.

Gráficos .png de convergencia y mapas de calor (en 2D y 3D).

Archivos .csv con los historiales de corridas y mejores individuos encontrados.