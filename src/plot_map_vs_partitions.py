# src/plot_tron_map_vs_mappart.py
import json, numpy as np, matplotlib.pyplot as plt, os

def load_hist(path):
    with open(path, "r") as f:
        j = json.load(f)
    H = j["history"]
    t = np.array([h["t_sec"] for h in H], dtype=float)
    fval = np.array([h["f"] for h in H], dtype=float)
    g = np.array([h["g_norm"] for h in H], dtype=float) if "g_norm" in H[0] else None
    return t, fval, g

def main():
    map_path = "logs/metrics_tron_map.json"
    part_path = "logs/metrics_tron_mappart.json"
    os.makedirs("graphics", exist_ok=True)

    t_map, f_map, g_map = load_hist(map_path)
    t_part, f_part, g_part = load_hist(part_path)

    # Figura 1: f(w) relativo vs tiempo (log10 como en el paper)
    f_star = min(f_map[-1], f_part[-1])
    rel_map  = np.log10(np.abs((f_map  - f_star) / max(abs(f_star), 1e-12) + 1e-12))
    rel_part = np.log10(np.abs((f_part - f_star) / max(abs(f_star), 1e-12) + 1e-12))

    plt.figure(figsize=(3.6, 3.2))
    plt.plot(t_map,  rel_map,  "-",  lw=2, label="map")
    plt.plot(t_part, rel_part, "--", lw=2, label="mapPartitions")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("log10( (f - f*) / |f*| )")
    plt.legend(fontsize=8)
    plt.title("webspam — TRON (map vs mapPartitions)")
    plt.tight_layout()
    plt.savefig("graphics/tron_webspam_f_rel_vs_time.png", dpi=200)
    plt.close()

    # Figura 2: ||g|| vs tiempo (escala log en Y para claridad)
    if g_map is not None and g_part is not None:
        plt.figure(figsize=(3.6, 3.2))
        plt.plot(t_map,  g_map,  "-",  lw=2, label="map")
        plt.plot(t_part, g_part, "--", lw=2, label="mapPartitions")
        plt.yscale("log")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("||grad||")
        plt.legend(fontsize=8)
        plt.title("webspam — TRON (norma del gradiente)")
        plt.tight_layout()
        plt.savefig("graphics/tron_webspam_grad_vs_time.png", dpi=200)
        plt.close()

    print("[OK] Gráficas guardadas en graphics/")

if __name__ == "__main__":
    main()
