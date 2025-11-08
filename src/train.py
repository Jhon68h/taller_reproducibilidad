# src/train.py
# CLI para entrenar modelos L2-regularizados (Logistic y L2-SVM) con TRON.
# Requisitos: numpy, y los módulos locales: tron, losses, io_libsvm, backend.

from __future__ import annotations
import argparse
import os
import sys
import time
import numpy as np

from tron import Tron
from losses import LogisticLoss, L2SVMLoss
from io_libsvm import load_libsvm_local, load_libsvm_rdd  # rdd opcional
from backend import LocalBackend  # SparkBackend (opcional)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrenamiento TRON (LR o L2-SVM) sobre datos LIBSVM."
    )
    p.add_argument("--mode", choices=["local", "spark"], default="local",
                   help="Ejecución local (NumPy) o Spark (opcional).")
    p.add_argument("--data", required=True, help="Ruta al archivo LIBSVM.")
    p.add_argument("--solver", choices=["lr", "l2svm"], default="lr",
                   help="Tipo de pérdida: lr (logística) o l2svm (hinge^2).")
    p.add_argument("--C", type=float, default=1.0, help="Regularización (C).")
    p.add_argument("--eps", type=float, default=1e-2, help="Tolerancia TRON.")
    p.add_argument("--bias", type=float, default=-1.0,
                   help="Bias: -1 desactiva; >=0 añade columna virtual con ese valor.")
    p.add_argument("--maxIter", type=int, default=100, help="Iteraciones máximas TRON.")
    # Parámetros Spark opcionales
    p.add_argument("--numPartitions", type=int, default=None,
                   help="Particiones al cargar en Spark.")
    p.add_argument("--numSlaves", type=int, default=None,
                   help="Coalesce antes de reducir en Spark (opcional).")
    p.add_argument("--seed", type=int, default=0, help="Semilla para reproducibilidad.")
    return p.parse_args(argv)


def select_loss(name: str):
    name = name.lower()
    if name == "lr":
        return LogisticLoss
    if name == "l2svm":
        return L2SVMLoss
    raise ValueError(f"solver desconocido: {name}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_local(args):
    print("[INFO] Cargando datos (local)...")
    X, y, n_features = load_libsvm_local(args.data, bias=args.bias)
    print(f"[INFO] n_samples={X.n_rows}  n_features={n_features}  bias={args.bias}")

    be = LocalBackend(X, y)
    LossCls = select_loss(args.solver)
    loss = LossCls(C=args.C, y=y, backend=be)

    w0 = np.zeros(n_features, dtype=np.float64)

    tron = Tron(
        f_fun=loss.f,
        g_fun=loss.grad,
        hv_fun=loss.hess_vec,
        eps=args.eps,
        max_iter=args.maxIter,
        verbose=True,
    )

    print("[INFO] Iniciando TRON...")
    t0 = time.time()
    w, info = tron.tron(w0)
    t1 = time.time()
    print(f"[INFO] Finalizado. iters={info['iters']}  f={info['f']:.6e}  "
          f"||g||={info['g_norm']:.3e}  tiempo={t1 - t0:.3f}s  motivo={info['reason']}")

    ensure_dir("./models")
    out_path = "./models/w.npy"
    np.save(out_path, w)
    print(f"[INFO] Modelo guardado en {out_path}")


def run_spark(args):
    # Estructura básica para futuras extensiones.
    # Si quieres habilitar Spark, implementa SparkBackend y adapta las operaciones
    # de f/grad/Hv a agregaciones por partición.
    raise NotImplementedError("Modo Spark no implementado aún en este script.")


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    np.set_printoptions(precision=4, suppress=True)

    if args.mode == "local":
        run_local(args)
    else:
        run_spark(args)


if __name__ == "__main__":
    main()
