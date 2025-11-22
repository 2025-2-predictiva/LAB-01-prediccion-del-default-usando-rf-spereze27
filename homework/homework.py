# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
from typing import Tuple, List, Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
INPUT_DIR = "files/input"
MODELS_DIR = "files/models"
OUTPUT_DIR = "files/output"

TRAIN_PATH = os.path.join(INPUT_DIR, "train_data.csv.zip")
TEST_PATH = os.path.join(INPUT_DIR, "test_data.csv.zip")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl.gz")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# Hiperparámetros ganadores
PARAM_GRID = {
    "model__n_estimators": [100],
    "model__max_depth": [None],
    "model__min_samples_split": [10],
    "model__min_samples_leaf": [4],
    "model__max_features": [None],
}

# ==========================================
# FUNCIONES DEL PROCESO
# ==========================================

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets de entrenamiento y prueba."""
    train = pd.read_csv(train_path, index_col=False, compression="zip")
    test = pd.read_csv(test_path, index_col=False, compression="zip")
    return train, test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza la limpieza y preprocesamiento básico de los datos."""
    df = df.copy()
    # Renombrar y eliminar columnas
    mapping = {"default payment next month": "default"}
    df = df.rename(columns=mapping)
    
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    
    # Eliminar nulos
    df = df.dropna()
    
    # Agrupar categorías de educación > 4 en 4 (others)
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        
    return df


def split_features_target(df: pd.DataFrame, target: str = "default") -> Tuple[pd.DataFrame, pd.Series]:
    """Separa las variables predictoras (X) de la variable objetivo (y)."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_pipeline() -> Pipeline:
    """Construye el pipeline de procesamiento y modelo."""
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    
    # Preprocesador: OneHot para categóricas, passthrough para el resto
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    # Pipeline completo
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Entrena el modelo usando GridSearchCV con los hiperparámetros definidos."""
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)
    return model


def save_model_compressed(model: Any, path: str) -> None:
    """Guarda el modelo serializado y comprimido con gzip."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def calculate_metrics(model: Any, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Tuple[Dict, Dict]:
    """Calcula métricas de desempeño y matriz de confusión."""
    y_pred = model.predict(X)

    # Métricas escalares
    metrics = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }

    # Matriz de confusión con estructura específica
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

    return metrics, cm_dict


def save_metrics_to_json(metrics_list: List[Dict], path: str) -> None:
    """Guarda la lista de métricas en un archivo JSON línea por línea."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")


def main():
    print("Iniciando proceso...")
    
    # 1. Carga
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    
    # 2. Limpieza
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    
    # 3. División X/y
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    
    # 4. Construcción y Entrenamiento
    pipeline = build_pipeline()
    grid_search = train_model(pipeline, X_train, y_train)
    
    # 5. Guardado del Modelo
    save_model_compressed(grid_search, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    
    # 6. Cálculo de Métricas
    train_metrics, train_cm = calculate_metrics(grid_search, X_train, y_train, "train")
    test_metrics, test_cm = calculate_metrics(grid_search, X_test, y_test, "test")
    
    # 7. Guardado de Métricas
    all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
    save_metrics_to_json(all_metrics, METRICS_PATH)
    print(f"Métricas guardadas en {METRICS_PATH}")
    print("Proceso finalizado con éxito.")

if __name__ == "__main__":
    main()