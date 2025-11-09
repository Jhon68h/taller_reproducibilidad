# src/bench_mllib_lr.py
from __future__ import annotations
import argparse, time, json, os, sys, platform, getpass
from datetime import datetime
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

# Para calcular el mismo objetivo que TRON:
from io_libsvm import load_libsvm_local
from backend import LocalBackend
from losses import LogisticLoss

def parse_args():
    ap = argparse.ArgumentParser("Benchmark Spark MLlib LogisticRegression (LBFGS)")
    ap.add_argument("--train", required=True, help="Ruta a train.svm (LIBSVM)")
    ap.add_argument("--test", required=True, help="Ruta a test.svm (LIBSVM)")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--maxIter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--fitIntercept", action="store_true")
    ap.add_argument("--out_weights", default="outputs/w_mllib_lr.npy")
    ap.add_argument("--out_json", default="logs/metrics_mllib_lr.json")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    spark = SparkSession.builder.appName("bench_mllib_lr").getOrCreate()

    # MLlib lee LIBSVM nativamente
    df_train = spark.read.format("libsvm").load(args.train)
    l = df_train.count()
    regParam = 1.0 / (args.C * l)  # mapeo LR: lambda = 1/(C*l)

    lr = LogisticRegression(
        maxIter=args.maxIter,
        regParam=regParam,
        elasticNetParam=0.0,
        standardization=False,
        tol=args.tol,
        fitIntercept=bool(args.fitIntercept),
    )

    t0 = time.time()
    model = lr.fit(df_train)
    t1 = time.time()
    train_time = t1 - t0

    # Pesos (sin intercepto). Si activas fitIntercept, puedes decidir si concatenarlo como bias.
    w = np.array(model.coefficients.toArray(), dtype=np.float64)
    np.save(args.out_weights, w)

    # Calcular MISMO objetivo LR sobre train con tu implementación
    X_tr, y_tr, n_tr = load_libsvm_local(args.train, bias=-1, n_features_force=w.size)
    be_tr = LocalBackend(X_tr, y_tr)
    loss_tr = LogisticLoss(C=args.C, y=y_tr, backend=be_tr)
    f_mllib = float(loss_tr.f(w))

    # Evaluación simple en test (accuracy/AUC) usando tu eval pipeline
    X_te, y_te, _ = load_libsvm_local(args.test, bias=-1, n_features_force=w.size)
    be_te = LocalBackend(X_te, y_te)
    margins = be_te.margin(w)
    y_pred = np.where(margins >= 0.0, 1.0, -1.0)
    acc = float((y_pred == y_te).mean())

    # AUC (método rápido tipo Mann–Whitney)
    y01 = (y_te > 0).astype(np.int64)
    pos = (y01 == 1); neg = ~pos
    if pos.sum() > 0 and neg.sum() > 0:
        order = np.argsort(margins)
        ranks = np.empty_like(order, dtype=np.int64); ranks[order] = np.arange(1, order.size + 1)
        sum_pos_ranks = ranks[pos].sum()
        U = sum_pos_ranks - pos.sum() * (pos.sum() + 1) / 2.0
        auc = float(U / (pos.sum() * neg.sum()))
    else:
        auc = float("nan")

    # JSON explícito
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
            "solver": "mllib_lr_lbfgs",
            "C": args.C,
            "regParam": regParam,
            "maxIter": args.maxIter,
            "tol": args.tol,
            "fitIntercept": bool(args.fitIntercept),
        },
        "summary": {
            "time_sec": train_time,
            "weights_path": os.path.abspath(args.out_weights),
            "f_train": f_mllib,                 # mismo objetivo LR que TRON
            "accuracy_test": acc,
            "roc_auc_test": auc,
        },
        "history": []  # MLlib no expone iteraciones fácilmente; puedes dejar vacío
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[MLlib] entrenamiento: {train_time:.3f}s; f_train={f_mllib:.6e}; "
          f"acc_test={acc:.4f}; auc_test={auc:.4f}")
    spark.stop()

if __name__ == "__main__":
    main()
