**#Water Potability — MLOps**

**##------- Descripción -------**

Proyecto de clasificación para predecir la potabilidad del agua mediante Random Forest. Se utiliza MLflow Tracking para registrar y comparar experimentos.

**##------- Tecnologías -------**

- Python
- Pandas
- Scikit-learn
- Matplotlib
- MLflow

**##------- Requisitos -------**

Python instalado y el archivo water_potability.csv ubicado en la misma carpeta que water_potability.py.

**##------- Instalación -------**

Instalar las dependencias con:

python -m pip install -r requirements.txt

**##------- Ejecución -------**

Ejemplo de entrenamiento:

python water_potability.py --n-estimators 100 --max-depth 50 --min-samples-leaf 1 --random-state 0

Los parámetros configurables son n_estimators, max_depth, min_samples_leaf y random_state.

**##------- Seguimiento con MLflow -------**

Para consultar la interfaz local, iniciar MLflow desde la carpeta del proyecto:

mlflow ui --backend-store-uri ./mlruns --port 5000

Abrir http://127.0.0.1:5000 en el navegador.

**##------- Métricas -------**

- Accuracy
- Recall de la clase potable (recall_potable)

**---- Artefactos ----**
MLflow registra el modelo y los artefactos de evaluación, incluida la matriz de confusión y el archivo de métricas.
