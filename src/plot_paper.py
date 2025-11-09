import json, numpy as np, matplotlib.pyplot as plt, os

def load_history(path):
    with open(path) as f:
        j = json.load(f)
    hist = j["history"]
    # Si no existe t_sec, usar iteraciones como eje de tiempo relativo
    if "t_sec" in hist[0]:
        t = np.array([h["t_sec"] for h in hist])
    else:
        t = np.arange(len(hist), dtype=float)
        t *= j["summary"].get("time_sec", 0) / max(1, len(hist) - 1)
    fval = np.array([h["f"] for h in hist])
    return t, fval


# Rutas de tus logs
aa_path = "logs/20251108_234148/metrics_l2svm_C100_eps1e-3_bias-1_rep3.json"
ca_path = "logs/metrics_l2svm_C100.0_eps0.001_bias-1_CA.json"

# Cargar historiales
t_aa, f_aa = load_history(aa_path)
t_ca, f_ca = load_history(ca_path)

# Normalizar tiempos si el arranque difiere
t_aa -= t_aa.min()
t_ca -= t_ca.min()

# Calcular valor mínimo común
f_star = min(f_aa[-1], f_ca[-1])
rel_aa = np.log10(np.abs((f_aa - f_star) / f_star))
rel_ca = np.log10(np.abs((f_ca - f_star) / f_star))

# Graficar
plt.figure(figsize=(4,3))
plt.plot(t_aa, rel_aa, 'r-', linewidth=2, label='AA (Array Approach)')
plt.plot(t_ca, rel_ca, 'b--', linewidth=2, label='CA (Class Approach)')
plt.xlabel("Training time (seconds)")
plt.ylabel("Relative function value difference (log10)")
plt.legend()
plt.title("(c) webspam — L2SVM, C=100, eps=1e-3")
plt.tight_layout()

os.makedirs("graphics", exist_ok=True)
plt.savefig("graphics/webspam_AA_CA_cmp.png", dpi=200)
plt.close()
print("[OK] Figura AA vs CA guardada en graphics/webspam_AA_CA_cmp.png")
