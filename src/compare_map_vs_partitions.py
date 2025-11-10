# compare_map_vs_partitions.py (versión corregida)
import json, numpy as np, matplotlib.pyplot as plt, os, argparse

def load_history(path):
    """Carga tiempos (t_sec) y valores f(w) desde un log JSON."""
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    hist = j.get("history", [])
    if not hist:
        raise RuntimeError(f"No hay 'history' en {path}")
    t = np.array([h.get("t_sec", np.nan) for h in hist], float)
    fval = np.array([h.get("f", np.nan) for h in hist], float)
    mask = np.isfinite(t) & np.isfinite(fval)
    return t[mask], fval[mask]

def rel_curve(fvals):
    """Calcula curva log10(|(f - f*) / f*|)."""
    f_star = np.nanmin(fvals)
    return np.log10(np.abs((fvals - f_star) / f_star + 1e-12)), f_star

def main():
    ap = argparse.ArgumentParser("Comparación real: map vs mapPartitions")
    ap.add_argument("--map_json", required=True, help="Log JSON (ejecución con map)")
    ap.add_argument("--mappart_json", required=True, help="Log JSON (ejecución con mapPartitions)")
    ap.add_argument("--title", default="webspam — map vs mapPartitions")
    ap.add_argument("--out", default="graphics/webspam_map_vs_partitions_real.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Cargar historiales reales
    t_map, f_map = load_history(args.map_json)
    t_mappart, f_mappart = load_history(args.mappart_json)

    # Normalizar tiempos desde cero
    t_map -= t_map.min()
    t_mappart -= t_mappart.min()

    # Curvas relativas
    f_star = min(np.nanmin(f_map), np.nanmin(f_mappart))
    rel_map = np.log10(np.abs((f_map - f_star) / f_star + 1e-12))
    rel_mappart = np.log10(np.abs((f_mappart - f_star) / f_star + 1e-12))

   # Graficar (tiempos reales, eje lineal)
    plt.figure(figsize=(3.2, 3.2))
    plt.plot(t_map, rel_map, 'r-', lw=2, label='map')
    plt.plot(t_mappart, rel_mappart, 'b--', lw=2, label='mapPartitions')
    plt.xlabel("Training time (seconds)")
    plt.ylabel("Relative function value difference (log10)")
    plt.legend()
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[OK] Figura guardada en {args.out}")


if __name__ == "__main__":
    main()
