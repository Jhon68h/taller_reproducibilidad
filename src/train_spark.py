# src/train_spark_tron.py
# Ejecuta TRON real (LR o L2-SVM) en Spark local, comparando
# dos implementaciones distribuidas para los productos X@v y X^T@r:
#   - "map":   vector denso por ejemplo y reduce
#   - "mappart": acumulación densa por partición
#
# Salidas:
#   logs/metrics_tron_map.json
#   logs/metrics_tron_mappart.json

from __future__ import annotations
import os, sys, time, json, argparse, getpass, platform
from datetime import datetime
import numpy as np

# --- asegurar import local de tus módulos (src/) ---
sys.path.append(os.path.abspath("src"))

from pyspark.sql import SparkSession
from tron import Tron
from losses import LogisticLoss, L2SVMLoss
from io_libsvm import load_libsvm_rdd

# ==========================================================
# SparkBackend distribuido (autocontenido en este script)
# ==========================================================
class SparkBackend:
    """
    Backend distribuido para TRON sobre un RDD de:
        (row_id:int, idx_list:list[int], val_list:list[float])
    Implementa margin, X_dot, Xt_dot con dos modos:
      - mode="map":       un vector denso por ejemplo y reduce
      - mode="mappart":   acumulación densa por partición
    """
    def __init__(self, rdd_rows, n_rows: int, n_features: int, mode: str = "mappart", num_slaves: int | None = None):
        assert mode in ("map", "mappart")
        self.rdd = rdd_rows  # (i, idx_list, val_list)
        self.n_rows = int(n_rows)
        self.n_features = int(n_features)
        self.mode = mode
        self.sc = rdd_rows.context
        self.num_slaves = num_slaves

    # ---------- util ----------
    def _assemble_vector_from_pairs(self, pairs):
        # pairs: list[(row_id, value)] -> np.ndarray[n_rows]
        m = np.empty(self.n_rows, dtype=np.float64)
        for i, v in pairs:
            m[int(i)] = float(v)
        return m

    # ---------- operaciones núcleo ----------
    def margin(self, w: np.ndarray) -> np.ndarray:
        """
        m = X @ w, en el orden de row_id (0..n_rows-1).
        """
        w = np.asarray(w, dtype=np.float64, copy=False)
        bc_w = self.sc.broadcast(w)
        rdd_local = self.rdd  # evitar capturar self en closures

        if self.mode == "map":
            # (i, idx, val) -> (i, <x_i, w>)
            pairs = rdd_local.map(
                lambda r: (r[0], float(np.dot(bc_w.value[r[1]], r[2])))
            ).collect()
        else:
            # mapPartitions para evitar muchos pares pequeños
            def part(it, wv):
                out = []
                for i, idx, val in it:
                    out.append((i, float(np.dot(wv[idx], val))))
                return iter(out)
            pairs = rdd_local.mapPartitions(lambda it: part(it, bc_w.value)).collect()

        bc_w.unpersist()
        return self._assemble_vector_from_pairs(pairs)

    def X_dot(self, v: np.ndarray) -> np.ndarray:
        """Xv = X @ v (sin duplicar lógica)."""
        return self.margin(v)

    def Xt_dot(self, r: np.ndarray) -> np.ndarray:
        """
        z = X^T @ r  (denso de tamaño n_features).
        r debe estar indexado por row_id.
        """
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 1 or r.size != self.n_rows:
            raise ValueError(f"r debe ser vector 1D de tamaño {self.n_rows}")

        bc_r = self.sc.broadcast(r)
        rdd_local = self.rdd
        n_features = self.n_features

        if self.mode == "map":
            # Cada fila produce un vector parcial denso y se reduce
            def contrib_row(row, r_vec, n_feat):
                i, idx, val = row  # idx, val son listas
                ri = r_vec[int(i)]
                z = np.zeros(n_feat, dtype=np.float64)
                for j, v in zip(idx, val):
                    z[j] += v * ri
                return z

            z = (
                rdd_local
                .map(lambda row: contrib_row(row, bc_r.value, n_features))
                .reduce(lambda a, b: a + b)
            )
        else:
            # Acumular contribuciones por partición
            def contrib_part(it, r_vec, n_feat):
                z = np.zeros(n_feat, dtype=np.float64)
                for i, idx, val in it:
                    ri = r_vec[int(i)]
                    for j, v in zip(idx, val):
                        z[j] += v * ri
                yield z

            rdd2 = rdd_local
            if self.num_slaves and self.num_slaves > 0:
                try:
                    rdd2 = rdd2.coalesce(self.num_slaves)
                except Exception:
                    pass

            z = (
                rdd2
                .mapPartitions(lambda it: contrib_part(it, bc_r.value, n_features))
                .reduce(lambda a, b: a + b)
            )

        bc_r.unpersist()
        return np.asarray(z, dtype=np.float64)


# ==========================================================
# utilidades
# ==========================================================
def select_loss(name: str):
    name = name.lower()
    if name == "lr":
        return LogisticLoss
    if name == "l2svm":
        return L2SVMLoss
    raise ValueError(f"solver desconocido: {name}")

def run_tron_distributed(
    spark, data_path: str, solver: str, C: float, eps: float,
    numPartitions: int | None, mode_backend: str, maxIter: int = 100,
    out_json: str = "logs/metrics.json", seed: int = 0
):
    """
    Entrena con TRON real sobre Spark usando SparkBackend (map o mappart).
    Guarda JSON con meta/config/summary/history (incluye t_sec).
    """
    sc = spark.sparkContext

    # Carga distribuida: (idx_np, val_np, y_float)
    rdd_raw, n_features = load_libsvm_rdd(sc, data_path, numPartitions=numPartitions, bias=-1.0)

    # row_id estable y conversión a listas nativas
    rdd_rows = (
        rdd_raw
        .zipWithIndex()
        .map(lambda rv: (int(rv[1]), rv[0][0].tolist(), rv[0][1].tolist()))
    )
    n_rows = rdd_rows.count()  # primero contar

    # Etiquetas locales alineadas por row_id
    y_pairs = (
        rdd_raw
        .zipWithIndex()
        .map(lambda rv: (int(rv[1]), float(rv[0][2])))
        .collect()
    )
    y = np.empty(n_rows, dtype=np.float64)
    for i, yi in y_pairs:
        y[int(i)] = yi

    # Backend distribuido
    be = SparkBackend(rdd_rows, n_rows=n_rows, n_features=n_features, mode=mode_backend)

    # Pérdida y TRON
    LossCls = select_loss(solver)
    loss = LossCls(C=C, y=y, backend=be)
    np.random.seed(seed)
    w0 = np.zeros(n_features, dtype=np.float64)

    start_wall = time.time()
    iter_history = []

    def cb(row):
        # row contiene f, g_norm, delta, etc.; añadimos tiempo acumulado
        row = dict(row)
        row["t_sec"] = time.time() - start_wall
        iter_history.append(row)

    tron = Tron(
        f_fun=loss.f, g_fun=loss.grad, hv_fun=loss.hess_vec,
        eps=eps, max_iter=maxIter, verbose=True, callback=cb
    )

    print(f"[INFO] Iniciando TRON — backend={mode_backend}  solver={solver}")
    t0 = time.time()
    w, info = tron.tron(w0)
    t1 = time.time()
    print(f"[INFO] Finalizado (backend={mode_backend}). iters={info['iters']}  "
          f"f={info['f']:.6e}  ||g||={info['g_norm']:.3e}  tiempo={t1 - t0:.3f}s")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "meta": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "user": getpass.getuser(),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "pyspark": spark.version,
            "mode": f"spark-{mode_backend}",
        },
        "config": {
            "data_path": os.path.abspath(data_path),
            "solver": solver,
            "C": C,
            "eps": eps,
            "bias": -1.0,
            "maxIter": maxIter,
            "n_samples": int(n_rows),
            "n_features": int(n_features),
            "numPartitions": numPartitions,
        },
        "summary": {
            "iters": int(info["iters"]),
            "final_f": float(info["f"]),
            "final_grad_norm": float(info["g_norm"]),
            "reason": info.get("reason", ""),
            "time_sec": float(t1 - t0),
            "weights_path": "",
        },
        "history": iter_history,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON guardado en {out_json}")


# ==========================================================
# CLI
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser("TRON distribuido en Spark: map vs mapPartitions")
    p.add_argument("--data", default="dataset/webspam_wc_normalized_unigram.svm")
    p.add_argument("--solver", choices=["lr", "l2svm"], default="l2svm")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--maxIter", type=int, default=100)
    p.add_argument("--numPartitions", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("TRON_Spark_Map_vs_MapPartitions")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    print(f"[INFO] Cargando datos desde {args.data} (partitions={args.numPartitions})")
    # Si empaquetas src/ como zip, distribúyelo:
    spark.sparkContext.addPyFile(os.path.abspath("src.zip"))

    # 1) Backend = map
    run_tron_distributed(
        spark, data_path=args.data, solver=args.solver, C=args.C, eps=args.eps,
        numPartitions=args.numPartitions, mode_backend="map", maxIter=args.maxIter,
        out_json="logs/metrics_tron_map.json", seed=args.seed
    )
    # 2) Backend = mapPartitions
    run_tron_distributed(
        spark, data_path=args.data, solver=args.solver, C=args.C, eps=args.eps,
        numPartitions=args.numPartitions, mode_backend="mappart", maxIter=args.maxIter,
        out_json="logs/metrics_tron_mappart.json", seed=args.seed
    )

    spark.stop()
    print("[FIN] Ejecución completada.")

if __name__ == "__main__":
    main()
