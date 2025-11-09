# src/bench_mllib_linsvc.py
from __future__ import annotations
import argparse, time, json, os, sys, platform, getpass
from datetime import datetime
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC

def parse_args():
    ap = argparse.ArgumentParser("Benchmark Spark MLlib LinearSVC (hinge)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--regParam", type=float, default=1.0)  # aquí MLlib usa regParam directo (L2)
    ap.add_argument("--maxIter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out_json", default="logs/metrics_mllib_linsvc.json")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)

    spark = SparkSession.builder.appName("bench_mllib_linsvc").getOrCreate()
    df_train = spark.read.format("libsvm").load(args.train)
    df_test = spark.read.format("libsvm").load(args.test)

    svm = LinearSVC(
        maxIter=args.maxIter,
        regParam=args.regParam,   # NO es C; no hay mapeo directo a hinge²
        tol=args.tol,
        standardization=False,
        fitIntercept=False
    )

    t0 = time.time()
    model = svm.fit(df_train)
    t1 = time.time()
    train_time = t1 - t0

    pred = model.transform(df_test).select("label", "rawPrediction")
    # Para accuracy: signo de rawPrediction[0]
    labs = np.array([r["label"] for r in pred.collect()], dtype=np.float64)
    scores = np.array([float(r["rawPrediction"][0]) for r in pred.collect()], dtype=np.float64)
    y_pred = np.where(scores >= 0.0, 1.0, -1.0)
    acc = float((y_pred == np.where(labs > 0, 1.0, -1.0)).mean())

    # AUC con scores
    y01 = (labs > 0).astype(np.int64)
    pos = (y01 == 1); neg = ~pos
    if pos.sum() > 0 and neg.sum() > 0:
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.int64); ranks[order] = np.arange(1, order.size + 1)
        sum_pos_ranks = ranks[pos].sum()
        U = sum_pos_ranks - pos.sum() * (pos.sum() + 1) / 2.0
        auc = float(U / (pos.sum() * neg.sum()))
    else:
        auc = float("nan")

    payload = {
        "meta": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "user": getpass.getuser(),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "pyspark": spark.version,
            "mode": "spark-mllib",
        },
        "config": {
            "train_path": os.path.abspath(args.train),
            "test_path": os.path.abspath(args.test),
            "solver": "mllib_linear_svc_hinge",
            "regParam": args.regParam,
            "maxIter": args.maxIter,
            "tol": args.tol,
            "fitIntercept": False,
        },
        "summary": {
            "time_sec": train_time,
            "accuracy_test": acc,
            "roc_auc_test": auc,
        },
        "history": []
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[MLlib LinearSVC] entrenamiento: {train_time:.3f}s; "
          f"acc_test={acc:.4f}; auc_test={auc:.4f}")
    spark.stop()

if __name__ == "__main__":
    main()
