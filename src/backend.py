# backend.py
# Backends para productos con X:
#  - LocalBackend: usa CSR ligero propio (csr.CSR)
#  - SparkBackend: ESQUELETO opcional (no implementado aquí)

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from csr import CSR


class LocalBackend:
    """
    Backend local sobre una matriz CSR densa en columnas (índices + valores).
    Expone:
      - margin(w)  = X @ w
      - X_dot(v)   = X @ v
      - Xt_dot(r)  = X^T @ r
    """

    def __init__(self, X: CSR, y: np.ndarray):
        if not isinstance(X, CSR):
            raise TypeError("X debe ser csr.CSR")
        self.X = X
        self.y = np.asarray(y, dtype=np.float64)
        if self.y.ndim != 1 or self.y.size != self.X.n_rows:
            raise ValueError("y debe ser vector 1D de tamaño igual a n_rows de X.")

    # --- Operaciones núcleo usadas por las pérdidas ---

    def margin(self, w: np.ndarray) -> np.ndarray:
        """m = X @ w"""
        return self.X.dot(w)

    def X_dot(self, v: np.ndarray) -> np.ndarray:
        """Xv = X @ v (sin duplicar lógica)"""
        return self.X.dot(v)

    def Xt_dot(self, r: np.ndarray) -> np.ndarray:
        """z = X^T @ r"""
        return self.X.Tdot(r)

    # --- Utilidades ---

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.X.n_rows, self.X.n_cols)

    @property
    def n_rows(self) -> int:
        return self.X.n_rows

    @property
    def n_cols(self) -> int:
        return self.X.n_cols


# -------------------------------------------------------------------------
# Esqueleto para Spark. Para habilitarlo:
#  1) Devuelve en io_libsvm_rdd un RDD de (row_id:int64, idx_np, val_np, y_float)
#     con row_id = zipWithIndex() estable.
#  2) Implementa margin/X_dot/Xt_dot usando broadcast(w) y broadcast(r),
#     y acumulación por partición en vectores densos.
# -------------------------------------------------------------------------


class SparkBackend:
    """
    Backend distribuido para TRON sobre un RDD de filas:
        (row_id:int, idx_list:list[int], val_list:list[float])
    Implementa margin, X_dot, Xt_dot con:
      - mode="map"       → un vector parcial por ejemplo
      - mode="mappart"   → acumulación densa por partición
    """

    def __init__(self, rdd, n_rows: int, n_features: int, mode: str = "map", num_slaves: int | None = None):
        assert mode in ("map", "mappart")
        self.rdd = rdd
        self.n_rows = int(n_rows)
        self.n_features = int(n_features)
        self.mode = mode
        self.num_slaves = num_slaves
        self.sc = rdd.context

    # ---------- operaciones núcleo ----------
    def margin(self, w: np.ndarray) -> np.ndarray:
        """
        m = X @ w, en el orden de row_id (0..n_rows-1).
        """
        bc_w = self.sc.broadcast(np.asarray(w, dtype=np.float64))
        rdd_local = self.rdd

        if self.mode == "map":
            # cada fila genera (row_id, margin)
            pairs = rdd_local.map(lambda r: (r[0], float(np.dot(bc_w.value[r[1]], r[2])))).collect()
        else:
            def part(it, wv):
                out = []
                for i, idx, val in it:
                    out.append((i, float(np.dot(wv[idx], val))))
                return iter(out)

            pairs = rdd_local.mapPartitions(lambda it: part(it, bc_w.value)).collect()

        bc_w.unpersist()

        # reconstruir vector completo
        m = np.empty(self.n_rows, dtype=np.float64)
        for i, v in pairs:
            m[int(i)] = v
        return m

    def X_dot(self, v: np.ndarray) -> np.ndarray:
        """X @ v (idéntico a margin)."""
        return self.margin(v)

    def Xt_dot(self, r: np.ndarray) -> np.ndarray:
        """
        z = X^T @ r (vector denso de tamaño n_features).
        """
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 1 or r.size != self.n_rows:
            raise ValueError(f"r debe ser vector 1D de tamaño {self.n_rows}")

        bc_r = self.sc.broadcast(r)
        rdd_local = self.rdd
        n_features = self.n_features

        if self.mode == "map":
            # Cada fila contribuye a z parcialmente
            def contrib_row(row, r_vec, n_feat):
                i, idx, val = row
                ri = r_vec[int(i)]
                z = np.zeros(n_feat, dtype=np.float64)
                for j, v in zip(idx, val):  # idx y val son listas
                    z[j] += v * ri
                return z

            z = (
                rdd_local
                .map(lambda row: contrib_row(row, bc_r.value, n_features))
                .reduce(lambda a, b: a + b)
            )

        else:
            # Acumular contribuciones por partición
            def contrib_part(it, r_vec, n_feat):
                z = np.zeros(n_feat, dtype=np.float64)
                for i, idx, val in it:
                    ri = r_vec[int(i)]
                    for j, v in zip(idx, val):
                        z[j] += v * ri
                yield z

            rdd2 = rdd_local
            if self.num_slaves and self.num_slaves > 0:
                try:
                    rdd2 = rdd2.coalesce(self.num_slaves)
                except Exception:
                    pass

            z = (
                rdd2
                .mapPartitions(lambda it: contrib_part(it, bc_r.value, n_features))
                .reduce(lambda a, b: a + b)
            )

        bc_r.unpersist()
        return np.asarray(z, dtype=np.float64)
