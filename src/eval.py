# src/eval.py
from __future__ import annotations
import argparse
import numpy as np
from io_libsvm import load_libsvm_local
from backend import LocalBackend

def roc_auc_from_scores(y_true, scores):
    y = (y_true > 0).astype(np.int64)
    pos = (y == 1)
    neg = ~pos
    n_pos = pos.sum(); n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    sum_pos_ranks = ranks[pos].sum()
    U = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))

def parse_args():
    ap = argparse.ArgumentParser("Evaluación con w.npy")
    ap.add_argument("--data", required=True, help="Ruta al .svm de evaluación")
    ap.add_argument("--weights", required=True, help="Ruta a outputs/w.npy")
    ap.add_argument("--bias", type=float, default=-1.0,
                    help="-1 sin bias; >=0 añade columna virtual con ese valor")
    ap.add_argument("--threshold", type=float, default=None,
                help="Umbral de decisión sobre el margen; por defecto 0.0")
    return ap.parse_args()

def main():
    args = parse_args()

    # Cargar pesos y forzar n_features al ancho de w
    w = np.load(args.weights)
    X, y, n_features = load_libsvm_local(
        args.data, bias=args.bias, n_features_force=int(w.size)
    )
    if w.size != n_features:
        raise ValueError(f"Dimensión de w ({w.size}) != n_features ({n_features}).")

    be = LocalBackend(X, y)
    margins = be.margin(w)
    thr = 0.0 if args.threshold is None else float(args.threshold)
    y_pred = np.where(margins >= thr, 1.0, -1.0)
    acc = (y_pred == y).mean()
    auc = roc_auc_from_scores(y, margins)
    print(f"[EVAL] n={y.size}  accuracy={acc:.4f}  roc_auc={auc:.4f}")

if __name__ == "__main__":
    main()
