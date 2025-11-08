# evaluation.py
# -*- coding: utf-8 -*-
"""
Comparación entre TRON distribuido y MLlib Logistic Regression,
ajustada al flujo del código original Spark-LIBLINEAR.
No hay división train/test. Se mide f(w) y tiempo de entrenamiento.
"""

import time
import json
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from data_loader import load_svm_data, get_spark_context
from distributed_tron import tron, _partition_obj_grad


# ===============================================================
#      FUNCIÓN PARA CALCULAR f(w) = C * loss + 0.5 * ||w||^2
# ===============================================================

def compute_objective(data_rdd, w, C, dim):
    sc = data_rdd.context
    w_b = sc.broadcast(w)
    loss_grad = data_rdd.mapPartitions(lambda part: _partition_obj_grad(part, w_b, C, dim))
    agg_loss, _ = loss_grad.reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    f = C * agg_loss + 0.5 * np.dot(w, w)
    return f


# ===============================================================
#                EVALUACIÓN DE TRON DISTRIBUIDO
# ===============================================================

def evaluate_tron(data_path, C=1.0, partitions=32, max_outer_iter=20, coalesce=None, eps=1e-4):
    sc = get_spark_context()
    data_rdd = load_svm_data(data_path, partitions=partitions)
    dim = data_rdd.map(lambda x: x[1].size).max() # type: ignore
    count = data_rdd.count()
    print(f"[TRON] Dataset: {count} instancias, dim={dim}, partitions={partitions}")

    t0 = time.time()
    w = tron(data_rdd, dim, C=C, eps=eps, max_outer_iter=max_outer_iter, coalesce_parts=coalesce)
    t1 = time.time()

    f = compute_objective(data_rdd, w, C, dim)
    print(f"[TRON] Tiempo total: {t1 - t0:.2f}s | f(w)={f:.6f}")

    np.save("../models/webspam_tron.npy", w)
    return {"method": "TRON", "time": t1 - t0, "f_obj": f}


# ===============================================================
#                EVALUACIÓN DE MLlib (Spark SGD)
# ===============================================================

def evaluate_mllib(data_path, C=1.0, partitions=32, max_iter=100):
    sc = get_spark_context()
    spark = SparkSession.builder.getOrCreate()

    data_rdd = load_svm_data(data_path, partitions=partitions)
    df = data_rdd.map(lambda xy: ((1.0 if xy[0] > 0 else 0.0),
                                  Vectors.dense(xy[1].toArray()))).toDF(["label", "features"])

    print(f"[MLlib] Dataset: {df.count()} instancias, partitions={partitions}")

    t0 = time.time()
    lr = LogisticRegression(
        maxIter=max_iter,
        regParam=1.0 / (C * df.count()),  # equivalente aproximado a C
        elasticNetParam=0.0,
        family="binomial"
    )
    model = lr.fit(df)
    t1 = time.time()

    print(f"[MLlib] Tiempo total: {t1 - t0:.2f}s")
    return {"method": "MLlib", "time": t1 - t0}


# ===============================================================
#                          MAIN
# ===============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comparación TRON vs MLlib (sin split, versión paper)")
    parser.add_argument("--data", type=str, required=True, help="Ruta del dataset en formato .svm")
    parser.add_argument("--C", type=float, default=1.0, help="Parámetro de regularización")
    parser.add_argument("--partitions", type=int, default=32, help="Número de particiones")
    parser.add_argument("--coalesce", type=int, default=None, help="Número de particiones tras coalesce")
    parser.add_argument("--max_outer_iter", type=int, default=20, help="Iteraciones TRON")
    parser.add_argument("--max_iter_mllib", type=int, default=100, help="Iteraciones MLlib")
    parser.add_argument("--eps", type=float, default=1e-4, help="Tolerancia de gradiente TRON")
    args = parser.parse_args()

    tron_result = evaluate_tron(
        args.data,
        C=args.C,
        partitions=args.partitions,
        max_outer_iter=args.max_outer_iter,
        coalesce=args.coalesce,
        eps=args.eps
    )

    mllib_result = evaluate_mllib(
        args.data,
        C=args.C,
        partitions=args.partitions,
        max_iter=args.max_iter_mllib
    )

    print("\n=== COMPARACIÓN FINAL ===")
    print(f"TRON  : {tron_result['time']:.2f}s | f(w)={tron_result['f_obj']:.6f}")
    print(f"MLlib : {mllib_result['time']:.2f}s")
    print("=========================")

    results = {"TRON": tron_result, "MLlib": mllib_result}
    with open("results_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[INFO] Resultados guardados en results_comparison.json")
