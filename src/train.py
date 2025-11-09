# src/train.py
from __future__ import annotations
import argparse, os, sys, time, json, platform, getpass
from datetime import datetime
import numpy as np

from tron import Tron
from losses import LogisticLoss, L2SVMLoss
from io_libsvm import load_libsvm_local
from backend import LocalBackend

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrenamiento TRON (LR o L2-SVM) sobre datos LIBSVM."
    )
    p.add_argument("--mode", choices=["local", "spark"], default="local")
    p.add_argument("--data", required=True)
    p.add_argument("--solver", choices=["lr", "l2svm"], default="lr")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--bias", type=float, default=-1.0)
    p.add_argument("--maxIter", type=int, default=100)
    p.add_argument("--numPartitions", type=int, default=None)
    p.add_argument("--numSlaves", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_json", type=str, default=None,
                   help="Ruta para guardar JSON con métricas; si no se pasa, se autogenera.")
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

    # Estructura de logging
    start_wall = time.time()
    start_iso = datetime.utcnow().isoformat() + "Z"
    iter_history = []  # se llena desde el callback

    def cb(row):
        # row ya trae: iter, f, g_norm, delta, actred, prered, ratio, cg, accepted
        now = time.time()
        row = dict(row)  # copia
        row["t_sec"] = now - start_wall  # tiempo acumulado
        iter_history.append(row)

    tron = Tron(
        f_fun=loss.f,
        g_fun=loss.grad,
        hv_fun=loss.hess_vec,
        eps=args.eps,
        max_iter=args.maxIter,
        verbose=True,
        callback=cb,
    )

    print("[INFO] Iniciando TRON...")
    t0 = time.time()
    w, info = tron.tron(w0)
    t1 = time.time()

    print(f"[INFO] Finalizado. iters={info['iters']}  f={info['f']:.6e}  "
          f"||g||={info['g_norm']:.3e}  tiempo={t1 - t0:.3f}s  motivo={info['reason']}")

    # Guardar pesos
    ensure_dir("./outputs")
    # Mantengo compat con tu script: models/ y outputs/
    ensure_dir("./models")
    np.save("./models/w.npy", w)     # como imprime tu log actual
    np.save("./outputs/w.npy", w)
    print("[INFO] Modelo guardado en ./models/w.npy")

    # Armar JSON con métricas explícitas
    ensure_dir("./logs")
    if args.log_json is None:
        tag = f"{args.solver}_C{args.C}_eps{args.eps}_bias{args.bias}".replace(".", "_")
        log_path = f"./logs/metrics_{tag}.json"
    else:
        log_path = args.log_json

    meta = {
        "timestamp_utc": start_iso,
        "user": getpass.getuser(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mode": "local",   # explícito: sin Spark
    }
    config = {
        "data_path": os.path.abspath(args.data),
        "solver": args.solver,
        "C": args.C,
        "eps": args.eps,
        "bias": args.bias,
        "maxIter": args.maxIter,
        "seed": args.seed,
        "n_samples": int(X.n_rows),
        "n_features": int(n_features),
    }
    summary = {
        "iters": int(info["iters"]),
        "final_f": float(info["f"]),
        "final_grad_norm": float(info["g_norm"]),
        "reason": info.get("reason", ""),
        "time_sec": float(t1 - t0),
        "start_time_epoch": float(t0),
        "end_time_epoch": float(t1),
        "weights_path": os.path.abspath("./models/w.npy"),
    }
    # Historial detallado por iteración (incluye tiempo acumulado)
    history = iter_history  # ya viene con t_sec

    payload = {
        "meta": meta,
        "config": config,
        "summary": summary,
        "history": history,
        # Campos útiles para reproducir
        "fields_description": {
            "history.iter": "Número de iteración externa (1..k)",
            "history.f": "Valor de la función objetivo en el punto aceptado",
            "history.g_norm": "Norma euclídea del gradiente en ese punto",
            "history.delta": "Radio de la región de confianza",
            "history.actred": "Reducción real f_old - f_new",
            "history.prered": "Reducción predicha por el modelo cuadrático",
            "history.ratio": "actred / prered",
            "history.cg": "Iteraciones de Conjugate Gradient en el subproblema",
            "history.accepted": "Si el paso fue aceptado",
            "history.t_sec": "Segundos acumulados desde el inicio del entrenamiento",
        }
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Métricas JSON guardadas en {log_path}")

def run_spark(args):
    raise NotImplementedError("Modo Spark no implementado en este script.")

def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    np.set_printoptions(precision=4, suppress=True)
    if args.mode != "local":
      raise NotImplementedError("Este script corre TRON en local (sin Spark).")
    run_local(args)

if __name__ == "__main__":
    main()
