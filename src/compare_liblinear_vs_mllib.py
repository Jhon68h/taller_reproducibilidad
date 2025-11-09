# compare_mllib_vs_tron.py
from __future__ import annotations
import argparse, os, time, json
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import when, col
from pyspark.ml.classification import LogisticRegression
# Reusamos tu stack local para evaluar f(w)
from io_libsvm import load_libsvm_local
from backend import LocalBackend
from losses import LogisticLoss

from pyspark.sql import SparkSession

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_tron_history(tron_json: str):
    with open(tron_json, "r", encoding="utf-8") as f:
        j = json.load(f)
    hist = j.get("history", [])
    if not hist:
        raise RuntimeError(f"No hay 'history' en {tron_json}")
    t = np.array([h.get("t_sec", np.nan) for h in hist], float)
    fvals = np.array([h.get("f", np.nan) for h in hist], float)
    # Limpieza
    mask = np.isfinite(t) & np.isfinite(fvals)
    return t[mask], fvals[mask]

def compute_objective_LR(weights: np.ndarray, C: float, data_path: str, bias: float):
    # Calcula f(w) = 0.5||w||^2 + C * sum log(1+exp(-y x⋅w)) en train
    X, y, _ = load_libsvm_local(data_path, bias=bias, n_features_force=weights.size)
    be = LocalBackend(X, y)
    loss = LogisticLoss(C=C, y=y, backend=be)
    return float(loss.f(weights))

def run_mllib_curve(train_path, C, bias, iters_grid, tol, fit_intercept=False):
    spark = SparkSession.builder.appName("mllib_curve_real").getOrCreate()
    df = spark.read.format("libsvm").load(train_path)
    df = df.withColumn("label", when(col("label") <= 0.0, 0.0).otherwise(1.0)).cache()
    l = df.count()
    regParam = 1.0 / (C * l)

    times, fvals, t_acc = [], [], 0.0
    for k in iters_grid:
        lr = LogisticRegression(
            maxIter=int(k),
            regParam=regParam,
            elasticNetParam=0.0,
            standardization=False,
            tol=tol,
            fitIntercept=False,   # <— sin intercepto
        )
        t0 = time.time(); model = lr.fit(df); t1 = time.time()
        t_acc += (t1 - t0)
        w = np.array(model.coefficients.toArray(), float)
        f_mllib = compute_objective_LR(w, C=C, data_path=train_path, bias=bias)  # bias=-1
        times.append(t_acc); fvals.append(f_mllib)
        print(f"[MLlib] maxIter={k:>4}  t_acum={t_acc:.3f}s  f={f_mllib:.6e}")

    df.unpersist(False); spark.stop()
    return np.array(times, float), np.array(fvals, float)

def relative_curve(t, fvals, f_star=None, normalize_by_start=True, eps=1e-12):
    """
    Devuelve y = log10( |(f - f*) / (f0 - f*)| ) como en el paper.
    Si normalize_by_start=False, usa |(f - f*) / |f*||.
    """
    if f_star is None:
        f_star = np.nanmin(fvals)
    f0 = fvals[0]
    if normalize_by_start:
        denom = max(abs(f0 - f_star), eps)
        rel = np.log10(np.clip(np.abs((fvals - f_star) / denom), eps, None))
    else:
        denom = max(abs(f_star), eps)
        rel = np.log10(np.clip(np.abs((fvals - f_star) / denom), eps, None))
    return rel, float(f_star)

def main():
    ap = argparse.ArgumentParser("Comparación real: TRON (Spark-LIBLINEAR Python) vs MLlib LR")
    ap.add_argument("--train", required=True, help="Ruta a train.svm (LIBSVM)")
    ap.add_argument("--tron_json", required=True, help="logs/metrics_lr_...json de train.py")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=-1.0)
    ap.add_argument("--tol", type=float, default=1e-6, help="tol de MLlib LR")
    ap.add_argument("--fitIntercept", action="store_true")
    ap.add_argument("--iters", default="1,2,3,5,10,20,50,100", help="grid de maxIter para MLlib")
    ap.add_argument("--logtime", action="store_true", help="eje X en log10(t)")
    ap.add_argument("--title", default="webspam — LR, C=1")
    ap.add_argument("--out", default="graphics/webspam_LIBLINEAR_vs_MLLIB_real.png")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out) or ".")
    iters_grid = [int(x) for x in args.iters.split(",") if x.strip()]

    # 1) Cargar historial real de TRON
    t_tron, f_tron = load_tron_history(args.tron_json)
    print(f"[TRON] puntos={t_tron.size}  t_max={t_tron[-1]:.3f}s  f_fin={f_tron[-1]:.6e}")

    # 2) Ejecutar MLlib varias veces y medir f(w) real en train
    t_mllib, f_mllib = run_mllib_curve(
        train_path=args.train,
        C=args.C,
        bias=args.bias,
        iters_grid=iters_grid,
        tol=args.tol,
        fit_intercept=args.fitIntercept,
    )

    # 3) Normalizar contra un f* común
    f_star = min(np.nanmin(f_tron), np.nanmin(f_mllib))
    rel_tron, _ = relative_curve(t_tron, f_tron, f_star=f_star, normalize_by_start=True)
    rel_mllib, _ = relative_curve(t_mllib, f_mllib, f_star=f_star, normalize_by_start=True)

    # 4) Trazar
    plt.figure(figsize=(3.4, 3.4))
    x_tron = np.log10(np.clip(t_tron, 1e-3, None)) if args.logtime else t_tron
    x_mllib = np.log10(np.clip(t_mllib, 1e-3, None)) if args.logtime else t_mllib

    plt.plot(x_tron, rel_tron, 'r-', lw=2, label='Spark LIBLINEAR (TRON)')
    plt.plot(x_mllib, rel_mllib, 'b--', lw=2, marker='o', ms=3.5, label='MLlib LR (LBFGS)')

    plt.xlabel("Training time (seconds)" + (" (log)" if args.logtime else ""))
    plt.ylabel("Relative objective value difference (log10)")
    plt.title(args.title)
    plt.ylim(-8, 2)
    plt.grid(alpha=0.2, linestyle="--", axis="y")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[OK] Figura guardada en {args.out}")

if __name__ == "__main__":
    main()
