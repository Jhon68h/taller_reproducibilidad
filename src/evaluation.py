# src/train_eval.py
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors

from data_loader import load_svm_data, get_spark_context
from distributed_tron import tron, predict_labels


# ======================================================
#        EVALUACIÓN DE TRON DISTRIBUIDO
# ======================================================

def evaluate_tron(data_path, C=1.0, partitions=32, max_outer_iter=20, coalesce=None, reuse_model=False):
    """
    Evalúa TRON (entrenando si es necesario o usando un modelo guardado).
    """
    sc = get_spark_context()
    data_rdd = load_svm_data(data_path, partitions=partitions)
    train_rdd, test_rdd = data_rdd.randomSplit([0.8, 0.2], seed=42)
    dim = train_rdd.map(lambda x: x[1].size).max()  # type: ignore
    model_path = "../models/webspam_tron.npy"

    # ------------------------------
    # Cargar o entrenar el modelo TRON
    # ------------------------------
    if reuse_model and os.path.exists(model_path):
        print(f"[TRON] Cargando modelo existente desde {model_path}")
        w = np.load(model_path)
        t_train = 0.0  # No se mide tiempo
    else:
        print(f"[TRON] Entrenando nuevo modelo (no se encontró {model_path})")
        t0 = time.time()
        w = tron(train_rdd, dim, C=C, max_outer_iter=max_outer_iter, coalesce_parts=coalesce)
        t_train = time.time() - t0
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        np.save(model_path, w)
        print(f"[TRON] Modelo guardado en {model_path}")

    # ------------------------------
    # Evaluación del modelo
    # ------------------------------
    acc, total, correct = predict_labels(test_rdd, w)
    print(f"[TRON] Tiempo: {t_train:.2f}s | Accuracy={acc*100:.2f}% ({correct}/{total})")
    return {"method": "TRON", "time": t_train, "accuracy": acc}


# ======================================================
#        EVALUACIÓN DE MLlib (LogisticRegression)
# ======================================================

def evaluate_mllib(data_path, C=1.0, partitions=32, max_iter=100):
    """Entrena y evalúa LogisticRegression de Spark MLlib."""
    sc = get_spark_context()
    spark = SparkSession.builder.getOrCreate()

    data_rdd = load_svm_data(data_path, partitions=partitions)
# Convertir etiquetas -1/+1 → 0/1 para MLlib
    df = data_rdd.map(lambda xy: ((1.0 if xy[0] > 0 else 0.0), Vectors.dense(xy[1].toArray()))).toDF(["label", "features"]) # type: ignore


    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    t0 = time.time()
    lr = LogisticRegression(
        maxIter=max_iter,
        regParam=1.0 / (C * train_df.count()),  # aproximación de C
        elasticNetParam=0.0,
        family="binomial"
    )
    model = lr.fit(train_df)
    t1 = time.time()

    preds = model.transform(test_df)
    correct = preds.filter(preds.label == preds.prediction).count()
    total = preds.count()
    acc = correct / total if total > 0 else 0.0

    print(f"[MLlib] Tiempo: {t1 - t0:.2f}s | Accuracy={acc*100:.2f}% ({correct}/{total})")
    return {"method": "MLlib", "time": t1 - t0, "accuracy": acc}


# ======================================================
#                MAIN
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comparación TRON vs MLlib")
    parser.add_argument("--data", type=str, required=True, help="Ruta del dataset en formato .svm")
    parser.add_argument("--C", type=float, default=1.0, help="Parámetro de regularización")
    parser.add_argument("--partitions", type=int, default=32, help="Número de particiones")
    parser.add_argument("--coalesce", type=int, default=None, help="Número de particiones tras coalesce")
    parser.add_argument("--max_outer_iter", type=int, default=20, help="Iteraciones TRON")
    parser.add_argument("--max_iter_mllib", type=int, default=100, help="Iteraciones MLlib")
    parser.add_argument("--reuse_model", action="store_true",
                        help="Usa modelo TRON previamente guardado (no reentrena)")
    args = parser.parse_args()

    # Ejecutar TRON (entrena o carga)
    tron_result = evaluate_tron(
        args.data,
        C=args.C,
        partitions=args.partitions,
        max_outer_iter=args.max_outer_iter,
        coalesce=args.coalesce,
        reuse_model=args.reuse_model
    )

    # Ejecutar MLlib siempre (para comparar rendimiento)
    mllib_result = evaluate_mllib(
        args.data,
        C=args.C,
        partitions=args.partitions,
        max_iter=args.max_iter_mllib
    )

    print("\n=== Comparación final ===")
    print(f"TRON : {tron_result['time']:.2f}s  | Acc={tron_result['accuracy']*100:.2f}%")
    print(f"MLlib: {mllib_result['time']:.2f}s  | Acc={mllib_result['accuracy']*100:.2f}%")

    # Guardar resultados
    with open("results_comparison.json", "w") as f:
        json.dump({"TRON": tron_result, "MLlib": mllib_result}, f, indent=2)
    print("[INFO] Resultados guardados en results_comparison.json")