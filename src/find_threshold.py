# src/find_threshold.py
from __future__ import annotations
import argparse, json, os, numpy as np
from io_libsvm import load_libsvm_local
from backend import LocalBackend

def f1_score(y, yhat):
    tp = np.sum((y==1) & (yhat==1)); fp = np.sum((y==-1) & (yhat==1))
    fn = np.sum((y==1) & (yhat==-1))
    prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def main():
    ap = argparse.ArgumentParser("Busca umbral óptimo en validación")
    ap.add_argument("--data", required=True, help="val.svm")
    ap.add_argument("--weights", required=True, help="models/w_*.npy")
    ap.add_argument("--bias", type=float, default=-1.0)
    ap.add_argument("--metric", choices=["acc","f1"], default="acc")
    ap.add_argument("--out", required=True, help="ruta JSON de salida")
    args = ap.parse_args()

    w = np.load(args.weights)
    X, y, _ = load_libsvm_local(args.data, bias=args.bias, n_features_force=w.size)
    be = LocalBackend(X, y)
    m = be.margin(w)

    # candidatos: todos los márgenes (estrategia ROC-like) + 0
    thr_cands = np.unique(m)
    thr_cands = np.concatenate([thr_cands, np.array([0.0])])
    best_thr, best_val = 0.0, -1.0

    for thr in thr_cands:
        yhat = np.where(m >= thr, 1.0, -1.0)
        if args.metric == "acc":
            val = float((yhat==y).mean())
        else:
            val = f1_score(y, yhat)
        if val > best_val:
            best_val, best_thr = val, float(thr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metric": args.metric, "best_threshold": best_thr, "best_score": best_val}, f, indent=2)
    print(f"[THR] metric={args.metric} best_threshold={best_thr:.6f} score={best_val:.4f}")

if __name__ == "__main__":
    main()
