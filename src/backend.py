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
# Esqueleto opcional para Spark. Para habilitarlo:
#  1) Devuelve en io_libsvm_rdd un RDD de (row_id:int64, idx_np, val_np, y_float)
#     con row_id = zipWithIndex() estable.
#  2) Implementa margin/X_dot/Xt_dot usando broadcast(w) y broadcast(r),
#     y acumulación por partición en vectores densos.
# -------------------------------------------------------------------------

class SparkBackend:
    """
    ESQUELETO: no implementado en este archivo.

    Requisitos para implementarlo correctamente:
      - RDD de registros: (row_id:int64, idx_np[int64], val_np[float64], y_float)
      - n_features: int
      - num_slaves: opcional, para coalesce antes del reduce

    Interfaz esperada:
      margin(w): np.ndarray con m[i] = <x_i, w> en el orden de row_id
      X_dot(v):  np.ndarray con (X @ v)[i]
      Xt_dot(r): np.ndarray con sum_i x_i * r[i]
    """

    def __init__(self, rdd, n_features: int, num_slaves: Optional[int] = None):
        self.rdd = rdd
        self.n_features = int(n_features)
        self.num_slaves = num_slaves
        raise NotImplementedError(
            "SparkBackend es un esqueleto. Implementa zipWithIndex en la carga y las "
            "operaciones margin/X_dot/Xt_dot con broadcast y reducciones por partición."
        )
