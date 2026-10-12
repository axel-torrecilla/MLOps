import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score,recall_score,confusion_matrix,ConfusionMatrixDisplay,)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---- Definicion de parametros desde terminal ----
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos para potabilidad del agua")

    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=0)

    return parser.parse_args()

# ---- Carga de Dataset ----
def load_data(data_path):
    df = pd.read_csv(data_path)

    if "Potability" not in df.columns:
        raise ValueError("El CSV debe contener la columna Potability.")

    if not set(df["Potability"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Potability debe contener únicamente 0 y 1.")

    df = df.dropna(subset=["Potability"])

    X = df.drop(columns="Potability")
    y = df["Potability"].astype(int)

    return X, y

# ---- Separar conjuntos de entrenamiento y prueba ----
def split_data(X, y, random_state):
    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y)

# ---- Crear modelo ----
def build_model(n_estimators, max_depth, min_samples_leaf, random_state):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1))])

# ---- Evaluar modelo ----
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "recall_potable": recall_score(
            y_test, predictions, zero_division=0)}

    return metrics, predictions

# ---- Registrar en MLFlow ----
def log_artifacts(y_test, predictions, metrics):
    os.makedirs("artifacts", exist_ok=True)

    metrics_path = os.path.join("artifacts", "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    mlflow.log_artifact(metrics_path, artifact_path="evaluation")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No potable", "Potable"])

    display.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusión - Random Forest")
    plt.tight_layout()

    cm_path = os.path.join("artifacts", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    mlflow.log_artifact(cm_path, artifact_path="evaluation")


def main():
    args = parse_args()

    mlflow.set_experiment("Water Potability - Random Forest")

    X, y = load_data('water_potability.csv')

    X_train, X_test, y_train, y_test = split_data(
        X, y, args.random_state)

    model = build_model(
        args.n_estimators,
        args.max_depth,
        args.min_samples_leaf,
        args.random_state,)

    with mlflow.start_run():
        # Registrar parámetros de la corrida
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", 0.30)
        mlflow.log_param("imputation_strategy", "median")

        # Entrenamiento y evaluación
        model.fit(X_train, y_train)
        metrics, predictions = evaluate_model(
            model, X_test, y_test)

        # Registrar métricas
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Registrar modelo y artefactos
        mlflow.sklearn.log_model(sk_model=model,artifact_path="model",serialization_format="cloudpickle")
        log_artifacts(y_test, predictions, metrics)

        print("Experimento:", "Water Potability - Random Forest")
        print("Parámetros:", {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state": args.random_state,
        })
        print("Métricas:", metrics)
        print("Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
