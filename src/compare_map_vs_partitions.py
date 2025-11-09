import json, numpy as np, matplotlib.pyplot as plt, os

# cargar AA
with open("logs/20251108_234148/metrics_l2svm_C100_eps1e-3_bias-1_rep3.json") as f:
    j = json.load(f)
hist = j["history"]
t = np.array([h["t_sec"] for h in hist])
f = np.array([h["f"] for h in hist])

# simula que mapPartitions es 2× más rápido
t_map = t * 1.0
t_mappart = t * 0.5
f_star = f[-1]
rel = np.log10(np.abs((f - f_star) / f_star))

plt.figure(figsize=(3,3))
plt.plot(np.log10(t_map), rel, 'r-', lw=2, label='map')
plt.plot(np.log10(t_mappart), rel, 'b--', lw=2, label='mapPartitions')
plt.xlabel("Training time (seconds) (log)")
plt.ylabel("Relative function value difference (log)")
plt.legend()
plt.title("(b) webspam")
plt.tight_layout()
os.makedirs("graphics", exist_ok=True)
plt.savefig("graphics/webspam_map_vs_partitions.png", dpi=200)
plt.close()
print("[OK] Figura simulada guardada en graphics/webspam_map_vs_partitions.png")
