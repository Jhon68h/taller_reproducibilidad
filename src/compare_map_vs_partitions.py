# src/compare_map_vs_mappartitions_real.py
import time, json, numpy as np, os
import sys, os
sys.path.append(os.path.abspath("src"))
from pyspark.sql import SparkSession
from io_libsvm import load_libsvm_rdd
spark = SparkSession.builder.master("local[*]").appName("Compare_Map_vs_MapPartitions_Real").getOrCreate()
sc = spark.sparkContext
sc.addPyFile(os.path.abspath("src.zip"))


# ==========================================================
# Funciones auxiliares
# ==========================================================

def compute_map(record, w_broadcast):
    """Simula operación local TRON usando map()."""
    idx, val, y = record
    w = w_broadcast.value
    s = float(np.dot(val, w[idx]))   # producto interno <x, w>
    return s * y


def compute_mappartition(iterator, w_broadcast):
    """Simula operación TRON por partición (mapPartitions)."""
    w = w_broadcast.value
    local_sum = 0.0
    for idx, val, y in iterator:
        s = float(np.dot(val, w[idx]))
        local_sum += s * y
    yield local_sum


# ==========================================================
# Ejecuciones
# ==========================================================

def run_map_version(rdd, n_features, iters=10):
    """Ejecuta la versión map()."""
    fvals, times = [], []
    t0 = time.time()
    w = np.random.randn(n_features)
    w_b = rdd.context.broadcast(w)
    for i in range(iters):
        s = rdd.map(lambda rec: compute_map(rec, w_b)).sum()
        fvals.append(s)
        times.append(time.time() - t0)
        print(f"[map] Iter {i+1:02d}  sum={s:.4f}  time={times[-1]:.3f}s")
    w_b.unpersist()
    return np.array(times), np.array(fvals)


def run_mappartitions_version(rdd, n_features, iters=10):
    """Ejecuta la versión mapPartitions()."""
    fvals, times = [], []
    t0 = time.time()
    w = np.random.randn(n_features)
    w_b = rdd.context.broadcast(w)
    for i in range(iters):
        s = rdd.mapPartitions(lambda it: compute_mappartition(it, w_b)).sum()
        fvals.append(s)
        times.append(time.time() - t0)
        print(f"[mapPartitions] Iter {i+1:02d}  sum={s:.4f}  time={times[-1]:.3f}s")
    w_b.unpersist()
    return np.array(times), np.array(fvals)


# ==========================================================
# Programa principal
# ==========================================================

if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("Compare_Map_vs_MapPartitions_Real")
        .getOrCreate()
    )
    sc = spark.sparkContext

    data_path = "dataset/webspam_wc_normalized_unigram.svm"
    rdd, n_features = load_libsvm_rdd(sc, data_path, numPartitions=8, bias=-1)
    rdd = rdd.cache()
    n_samples = rdd.count()
    print(f"[INFO] Dataset cargado: {n_samples} muestras, {n_features} características")

    os.makedirs("logs", exist_ok=True)

    # -------------------------
    # 1. Ejecutar versión map()
    # -------------------------
    print("\n=== Ejecutando versión map() ===")
    t_map, f_map = run_map_version(rdd, n_features, iters=10)
    with open("logs/map_run.json", "w") as f:
        json.dump({"t_map": t_map.tolist(), "f_map": f_map.tolist()}, f, indent=2)
    print("[OK] Guardado: logs/map_run.json")

    # -------------------------------
    # 2. Ejecutar versión mapPartitions()
    # -------------------------------
    print("\n=== Ejecutando versión mapPartitions() ===")
    t_mappart, f_mappart = run_mappartitions_version(rdd, n_features, iters=10)
    with open("logs/mappart_run.json", "w") as f:
        json.dump({"t_mappart": t_mappart.tolist(), "f_mappart": f_mappart.tolist()}, f, indent=2)
    print("[OK] Guardado: logs/mappart_run.json")

    spark.stop()
    print("\n[FIN] Ejecución completada.")
