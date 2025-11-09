# src/split_libsvm.py
from __future__ import annotations
import argparse, random

def parse_args():
    ap = argparse.ArgumentParser("Split LIBSVM en train/test")
    ap.add_argument("--data", required=True, help="Ruta al .svm")
    ap.add_argument("--train_out", required=True, help="Salida train.svm")
    ap.add_argument("--test_out", required=True, help="Salida test.svm")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if ln.strip() and not ln.strip().startswith("#")]
    rng.shuffle(lines)
    n = len(lines)
    k = int(args.train_ratio * n)
    with open(args.train_out, "w", encoding="utf-8") as ft:
        ft.writelines(lines[:k])
    with open(args.test_out, "w", encoding="utf-8") as fv:
        fv.writelines(lines[k:])
    print(f"[INFO] split -> train:{k}  test:{n-k}")

if __name__ == "__main__":
    main()
