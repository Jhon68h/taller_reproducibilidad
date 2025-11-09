from __future__ import annotations
import argparse, os, json, glob, sys, re
import numpy as np
import matplotlib.pyplot as plt
from io_libsvm import load_libsvm_local
from backend import LocalBackend

def auc_from_scores(y, s):
    y01 = (y > 0).astype(np.int64)
    pos = (y01 == 1)
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, s.size + 1)
    U = ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0
    return float(U / (pos.sum() * neg.sum()))

def parse_args():
    ap = argparse.ArgumentParser("Plot métricas (agrupado y con umbral LR)")
    ap.add_argument("--logs_glob", default="logs/*/metrics_*.json")
    ap.add_argument("--outdir", default="graphics")
    ap.add_argument("--test", default="data/test.svm")
    return ap.parse_args()

def load_runs(pattern):
    paths = sorted(glob.glob(pattern))
    runs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        tag = os.path.splitext(os.path.basename(p))[0].replace("metrics_", "")
        runs.append((p, tag, j))
    return runs

def tag_base(tag):
    return re.sub(r"_rep\d+$", "", tag)

def solver_from_tag(tag, j):
    s = j.get("config", {}).get("solver", "")
    if s: return s
    if tag.startswith("lr_"): return "lr"
    if tag.startswith("l2svm_"): return "l2svm"
    return "?"

def try_load_threshold(json_path, tag):
    d = os.path.dirname(json_path)
    thr_path = os.path.join(d, f"threshold_{tag}.json")
    if os.path.isfile(thr_path):
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                return float(json.load(f)["best_threshold"])
        except Exception:
            return None
    return None

def grouped_bar(ax, names, means, stds, ylabel, title):
    idx = np.arange(len(names))
    ax.bar(idx, means, yerr=stds, capsize=3)
    # Limpieza de etiquetas
    short = [re.sub(r"^(lr_|l2svm_)", "", n) for n in names]
    short = [re.sub(r"_bias-1$", "", n) for n in short]
    ax.set_xticks(idx)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = load_runs(args.logs_glob)
    if not runs:
        print(f"No hay JSONs con {args.logs_glob}", file=sys.stderr)
        return

    # --- Curvas f(t) y ||g||(t) por solver ---
    for which in ["lr", "l2svm"]:
        for key in ["f", "g_norm"]:
            plt.figure()
            had = False
            for jpath, tag, data in runs:
                if solver_from_tag(tag, data) != which: continue
                hist = data.get("history", [])
                t = np.array([h.get("t_sec", np.nan) for h in hist], float)
                val = np.array([h.get(key, np.nan) for h in hist], float)
                if t.size == 0 or np.all(np.isnan(t)): continue
                plt.plot(t, val, label=tag); had = True
            if had:
                plt.xlabel("Tiempo (s)")
                plt.ylabel("f(w)" if key == "f" else "||g|| (log)")
                if key == "g_norm":
                    plt.yscale("log")
                plt.title(f"Convergencia {'f(w)' if key=='f' else '||g||'} vs tiempo — {which}")
                plt.legend(fontsize=6, ncol=2)
                plt.tight_layout()
                outname = f"convergence_{'f' if key=='f' else 'grad'}_{which}.png"
                plt.savefig(os.path.join(args.outdir, outname), dpi=160)
            plt.close()

    # --- Agrupar por configuración base (sin rep) y calcular medias ---
    groups = {}
    for jpath, tag, data in runs:
        base = tag_base(tag)
        groups.setdefault(base, []).append((jpath, tag, data))

    def order_key(base):
        m = re.search(r"C([0-9e.-]+)", base)
        return float(m.group(1)) if m else 0.0

    # Tiempo + f_final agrupados (separar por solver)
    for which in ["lr", "l2svm"]:
        names, t_mean, t_std, f_mean, f_std = [], [], [], [], []
        for base, lst in sorted(groups.items(), key=lambda kv: order_key(kv[0])):
            if solver_from_tag(base, lst[0][2]) != which: continue
            times = [float(x[2]["summary"].get("time_sec", np.nan)) for x in lst]
            finals = [float(x[2]["summary"].get("final_f", np.nan)) for x in lst]
            names.append(base)
            t_mean.append(np.nanmean(times)); t_std.append(np.nanstd(times))
            f_mean.append(np.nanmean(finals)); f_std.append(np.nanstd(finals))

        if names:
            fig, ax = plt.subplots(figsize=(8, 4))
            grouped_bar(ax, names, t_mean, t_std, "Tiempo (s)", f"Tiempo total — {which}")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"time_total_{which}.png"), dpi=160); plt.close()

            fig, ax = plt.subplots(figsize=(8, 4))
            grouped_bar(ax, names, f_mean, f_std, "f(w) final", f"Objetivo final — {which}")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"final_f_{which}.png"), dpi=160); plt.close()

    # --- Accuracy/AUC con umbral de validación (si existe) ---
    if args.test and os.path.isfile(args.test):
        names, acc_mean, acc_std, auc_mean, auc_std = [], [], [], [], []
        for base, lst in sorted(groups.items(), key=lambda kv: order_key(kv[0])):
            accs, aucs = [], []
            for jpath, tag, data in lst:
                thr = try_load_threshold(jpath, tag)
                bias = float(data.get("config", {}).get("bias", -1.0))
                w_path = data.get("summary", {}).get("weights_path", "")
                run_dir = os.path.basename(os.path.dirname(jpath))
                alt = os.path.join("models", run_dir, f"w_{tag}.npy")
                if os.path.isfile(alt): w_path = alt
                if not w_path or not os.path.isfile(w_path):
                    continue
                w = np.load(w_path)
                X, y, _ = load_libsvm_local(args.test, bias=bias, n_features_force=w.size)
                be = LocalBackend(X, y)
                m = be.margin(w)
                if thr is None: thr = 0.0
                yhat = np.where(m >= thr, 1.0, -1.0)
                accs.append(float((yhat == y).mean()))
                aucs.append(auc_from_scores(y, m))
            if accs:
                names.append(base)
                acc_mean.append(float(np.mean(accs))); acc_std.append(float(np.std(accs)))
                auc_mean.append(float(np.mean(aucs))); auc_std.append(float(np.std(aucs)))

        if names:
            fig, ax = plt.subplots(figsize=(8, 4))
            grouped_bar(ax, names, acc_mean, acc_std, "Accuracy (test)", "Accuracy (media ±σ, umbral val si disponible)")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "accuracy_test_grouped.png"), dpi=160); plt.close()

            fig, ax = plt.subplots(figsize=(8, 4))
            grouped_bar(ax, names, auc_mean, auc_std, "ROC-AUC (test)", "AUC (media ±σ)")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "auc_test_grouped.png"), dpi=160); plt.close()

    print(f"[OK] Figuras ordenadas y limpias en {args.outdir}")

if __name__ == "__main__":
    main()
