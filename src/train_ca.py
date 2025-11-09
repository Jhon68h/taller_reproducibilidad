import time, json, os, platform, getpass, sys, numpy as np
from datetime import datetime
from io_libsvm import load_libsvm_local
from tron import Tron
from losses import L2SVMLoss
from backend import LocalBackend

# --- Representación tipo CA ---
class Feature:
    __slots__ = ("idx","val")
    def __init__(self, idx, val):
        self.idx = idx
        self.val = val

def convert_to_CA(X):
    """Convierte CSR -> lista de listas de objetos Feature."""
    features = []
    for i in range(X.n_rows):
        s, e = X.indptr[i], X.indptr[i+1]
        feats = [Feature(int(j), float(v)) for j,v in zip(X.indices[s:e], X.data[s:e])]
        features.append(feats)
    return features

# --- Backend alternativo que itera sobre objetos Feature ---
class ClassBackend:
    """
    Backend tipo 'Class Approach': usa listas de objetos Feature.
    Cada muestra es una lista de Feature(idx, val).
    """
    def __init__(self, features, y, n_features):
        self.features = features
        self.y = np.asarray(y, np.float64)
        self.n_features = int(n_features)
        self.n_rows = len(features)

    def margin(self, w):
        """m = X @ w"""
        m = np.zeros(self.n_rows)
        for i, feats in enumerate(self.features):
            acc = 0.0
            for f in feats:
                acc += f.val * w[f.idx]
            m[i] = acc
        return m

    def X_dot(self, v):
        """X @ v"""
        return self.margin(v)

    def Xt_dot(self, r):
        """X^T @ r"""
        z = np.zeros(self.n_features)
        for i, feats in enumerate(self.features):
            ri = r[i]
            if ri == 0.0:
                continue
            for f in feats:
                z[f.idx] += f.val * ri
        return z

# --- Entrenamiento usando la versión CA ---
data_path = "data/train.svm"
bias = -1
C = 100.0
eps = 1e-3

X, y, n_features = load_libsvm_local(data_path, bias=bias)
features = convert_to_CA(X)
be = ClassBackend(features, y, n_features)
loss = L2SVMLoss(C=C, y=y, backend=be)
w0 = np.zeros(n_features)

start = time.time()
tron = Tron(loss.f, loss.grad, loss.hess_vec, eps=eps, max_iter=100, verbose=True)
w, info = tron.tron(w0)
elapsed = time.time() - start

os.makedirs("logs", exist_ok=True)
tag = f"l2svm_C{C}_eps{eps}_bias{bias}_CA"
log_path = f"logs/metrics_{tag}.json"
summary = {"time_sec": elapsed, "final_f": info['f'], "final_grad_norm": info['g_norm']}
with open(log_path, "w") as f:
    json.dump({"config":{"solver":"l2svm","approach":"CA"}, "summary":summary,
               "history":info["history"]}, f, indent=2)
print(f"[OK] Entrenamiento CA completado ({elapsed:.2f}s) -> {log_path}")
