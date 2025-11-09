# src/plot_metrics.py (reemplazo mejorado)
from __future__ import annotations
import argparse, os, json, glob, sys, re
import numpy as np
import matplotlib.pyplot as plt
from io_libsvm import load_libsvm_local
from backend import LocalBackend

def auc_from_scores(y, s):
    y01=(y>0).astype(np.int64); pos=(y01==1); neg=~pos
    if pos.sum()==0 or neg.sum()==0: return float("nan")
    order=np.argsort(s); ranks=np.empty_like(order); ranks[order]=np.arange(1, s.size+1)
    U=ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2.0
    return float(U/(pos.sum()*neg.sum()))

def parse_args():
    ap=argparse.ArgumentParser("Plot métricas (agrupado y con umbral LR)")
    ap.add_argument("--logs_glob", default="logs/*/metrics_*.json")
    ap.add_argument("--outdir", default="graphics")
    ap.add_argument("--test", default="data/test.svm")
    return ap.parse_args()

def load_runs(pattern):
    paths=sorted(glob.glob(pattern))
    runs=[]
    for p in paths:
        with open(p,"r",encoding="utf-8") as f:
            j=json.load(f)
        tag=os.path.splitext(os.path.basename(p))[0].replace("metrics_","")
        runs.append((p,tag,j))
    return runs

def tag_base(tag):
    # quita _repX
    return re.sub(r"_rep\d+$","",tag)

def solver_from_tag(tag, j):
    s=j.get("config",{}).get("solver","")
    if s: return s
    # fallback por nombre
    if tag.startswith("lr_"): return "lr"
    if tag.startswith("l2svm_"): return "l2svm"
    return "?"

def try_load_threshold(json_path, tag):
    # busca un threshold_<TAG>.json en la misma carpeta del metrics
    d=os.path.dirname(json_path)
    thr_path=os.path.join(d, f"threshold_{tag}.json")
    if os.path.isfile(thr_path):
        try:
            with open(thr_path,"r",encoding="utf-8") as f:
                return float(json.load(f)["best_threshold"])
        except Exception:
            return None
    return None

def grouped_bar(ax, names, means, stds, ylabel, title):
    idx=np.arange(len(names))
    ax.bar(idx, means, yerr=stds, capsize=3)
    ax.set_xticks(idx); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

def main():
    args=parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs=load_runs(args.logs_glob)
    if not runs:
        print(f"No hay JSONs con {args.logs_glob}", file=sys.stderr); return

    # --- Curvas f(t) y ||g||(t) por solver ---
    for which in ["lr","l2svm"]:
        plt.figure()
        had=False
        for jpath, tag, data in runs:
            if solver_from_tag(tag, data)!=which: continue
            hist=data.get("history",[])
            t=np.array([h.get("t_sec", np.nan) for h in hist], float)
            f=np.array([h.get("f", np.nan) for h in hist], float)
            if t.size==0 or np.all(np.isnan(t)): continue
            plt.plot(t, f, label=tag); had=True
        if had:
            plt.xlabel("Tiempo (s)"); plt.ylabel("f(w)"); plt.title(f"Convergencia f(w) vs tiempo — {which}")
            plt.legend(fontsize=8); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"convergence_f_{which}.png"), dpi=160)
        plt.close()

        plt.figure()
        had=False
        for jpath, tag, data in runs:
            if solver_from_tag(tag, data)!=which: continue
            hist=data.get("history",[])
            t=np.array([h.get("t_sec", np.nan) for h in hist], float)
            g=np.array([h.get("g_norm", np.nan) for h in hist], float)
            if t.size==0 or np.all(np.isnan(t)): continue
            plt.semilogy(t, g, label=tag); had=True
        if had:
            plt.xlabel("Tiempo (s)"); plt.ylabel("||g|| (log)"); plt.title(f"Convergencia ||g|| vs tiempo — {which}")
            plt.legend(fontsize=8); plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"convergence_grad_{which}.png"), dpi=160)
        plt.close()

    # --- Agrupar por configuración base (sin rep) y calcular medias ---
    groups={}
    for jpath, tag, data in runs:
        base=tag_base(tag)
        groups.setdefault(base, []).append((jpath, tag, data))

    # Tiempo + f_final agrupados (separar por solver)
    for which in ["lr","l2svm"]:
        names=[]; t_mean=[]; t_std=[]; f_mean=[]; f_std=[]
        for base, lst in sorted(groups.items()):
            # filtrar por solver
            if solver_from_tag(base, lst[0][2])!=which: continue
            times=[float(x[2]["summary"].get("time_sec", np.nan)) for x in lst]
            finals=[float(x[2]["summary"].get("final_f", np.nan)) for x in lst]
            names.append(base); t_mean.append(np.nanmean(times)); t_std.append(np.nanstd(times))
            f_mean.append(np.nanmean(finals)); f_std.append(np.nanstd(finals))

        if names:
            fig,ax=plt.subplots(1,1,figsize=(8,4))
            grouped_bar(ax, names, t_mean, t_std, "Tiempo (s)", f"Tiempo total — {which}")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"time_total_{which}.png"), dpi=160); plt.close()

            fig,ax=plt.subplots(1,1,figsize=(8,4))
            grouped_bar(ax, names, f_mean, f_std, "f(w) final", f"Objetivo final — {which}")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"final_f_{which}.png"), dpi=160); plt.close()

    # --- Accuracy/AUC con umbral de validación (si existe) ---
    if args.test and os.path.isfile(args.test):
        names=[]; acc_mean=[]; acc_std=[]; auc_mean=[]; auc_std=[]
        for base, lst in sorted(groups.items()):
            # pesos: preferimos models/<run>/w_<tag>.npy; si no, summary.weights_path
            accs=[]; aucs=[]
            for jpath, tag, data in lst:
                # umbral: si hay threshold_<tag>.json, úsalo (solo LR lo tendrá)
                thr = try_load_threshold(jpath, tag)
                bias=float(data.get("config",{}).get("bias",-1.0))
                # localizar pesos
                w_path=data.get("summary",{}).get("weights_path","")
                # buscar también en la carpeta de modelos del mismo RUN_TAG
                run_dir=os.path.basename(os.path.dirname(jpath))  # p.ej. 20251108_223440
                alt=os.path.join("models", run_dir, f"w_{tag}.npy")
                if os.path.isfile(alt): w_path=alt
                if not w_path or not os.path.isfile(w_path):
                    continue
                w=np.load(w_path)
                X,y,_=load_libsvm_local(args.test, bias=bias, n_features_force=w.size)
                be=LocalBackend(X,y); m=be.margin(w)
                if thr is None: thr=0.0
                yhat=np.where(m>=thr, 1.0, -1.0)
                accs.append(float((yhat==y).mean()))
                aucs.append(auc_from_scores(y,m))
            if accs:
                names.append(base)
                acc_mean.append(float(np.mean(accs))); acc_std.append(float(np.std(accs)))
                auc_mean.append(float(np.mean(aucs))); auc_std.append(float(np.std(aucs)))

        if names:
            fig,ax=plt.subplots(1,1,figsize=(8,4))
            grouped_bar(ax, names, acc_mean, acc_std, "Accuracy (test)", "Accuracy (media ±σ, umbral val si disponible)")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "accuracy_test_grouped.png"), dpi=160); plt.close()

            fig,ax=plt.subplots(1,1,figsize=(8,4))
            grouped_bar(ax, names, auc_mean, auc_std, "ROC-AUC (test)", "AUC (media ±σ)")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "auc_test_grouped.png"), dpi=160); plt.close()

    print(f"[OK] Figuras en {args.outdir}")
