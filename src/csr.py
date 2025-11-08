# csr.py
# Implementación ligera de una matriz dispersa en formato CSR.
# Funcionalidad: productos X @ v y X^T @ r con dtype float64.

from __future__ import annotations
import numpy as np
from typing import Tuple

Array = np.ndarray


class CSR:
    """
    Matriz dispersa en formato CSR (Compressed Sparse Row).

    Atributos:
      indptr  : np.int64 [n_rows+1]
      indices : np.int64 [nnz]
      data    : np.float64 [nnz]
      n_rows  : int
      n_cols  : int
      nnz     : int

    Convenciones:
      - Los índices de columna son 0-based.
      - No se requiere ordenamiento estricto por fila, pero es recomendable.
    """

    def __init__(self, indptr: Array, indices: Array, data: Array, n_cols: int):
        if indptr.ndim != 1 or indices.ndim != 1 or data.ndim != 1:
            raise ValueError("indptr, indices y data deben ser 1D.")
        if indptr.size == 0:
            raise ValueError("indptr no puede ser vacío.")
        if indices.size != data.size:
            raise ValueError("indices y data deben tener el mismo tamaño.")
        if indptr[-1] != indices.size:
            raise ValueError("indptr[-1] debe ser igual a nnz (len(indices)).")

        self.indptr = indptr.astype(np.int64, copy=False)
        self.indices = indices.astype(np.int64, copy=False)
        self.data = data.astype(np.float64, copy=False)
        self.n_rows = int(self.indptr.size - 1)
        self.n_cols = int(n_cols)
        self.nnz = int(self.indices.size)

        if np.any(self.indices < 0) or np.any(self.indices >= self.n_cols):
            raise ValueError("indices fuera de rango de columnas.")

    # --------------------- productos básicos --------------------- #

    def dot(self, v: Array) -> Array:
        """
        y = X @ v, con v de tamaño n_cols.
        Devuelve y de tamaño n_rows.
        """
        v = np.asarray(v, dtype=np.float64)
        if v.ndim != 1 or v.size != self.n_cols:
            raise ValueError(f"v debe ser vector 1D de tamaño {self.n_cols}.")

        y = np.zeros(self.n_rows, dtype=np.float64)
        # Recorre filas
        for i in range(self.n_rows):
            start, end = self.indptr[i], self.indptr[i + 1]
            if start == end:
                continue
            idx = self.indices[start:end]
            val = self.data[start:end]
            # Producto fila i por v
            y[i] = val @ v[idx]
        return y

    def Tdot(self, r: Array) -> Array:
        """
        z = X^T @ r, con r de tamaño n_rows.
        Devuelve z de tamaño n_cols (denso).
        """
        r = np.asarray(r, dtype=np.float64)
        if r.ndim != 1 or r.size != self.n_rows:
            raise ValueError(f"r debe ser vector 1D de tamaño {self.n_rows}.")

        z = np.zeros(self.n_cols, dtype=np.float64)
        # Recorre filas y acumula en columnas
        for i in range(self.n_rows):
            ri = r[i]
            if ri == 0.0:
                continue
            start, end = self.indptr[i], self.indptr[i + 1]
            if start == end:
                continue
            idx = self.indices[start:end]
            val = self.data[start:end]
            # z[idx] += val * r[i]
            z[idx] += val * ri
        return z

    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)

    def to_dense(self) -> Array:
        """
        Solo para depuración. Construye una matriz densa (cuidado con memoria).
        """
        M = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        for i in range(self.n_rows):
            s, e = self.indptr[i], self.indptr[i + 1]
            if s == e:
                continue
            M[i, self.indices[s:e]] = self.data[s:e]
        return M

    def __matmul__(self, v: Array) -> Array:
        """Permite usar operador @: X @ v."""
        return self.dot(v)

    def T(self) -> "CSR_T":
        """Devuelve un envoltorio ligero para aplicar X^T @ r vía operador @."""
        return CSR_T(self)


class CSR_T:
    """
    Envoltorio para permitir (X.T) @ r con el operador @.
    Solo implementa __matmul__ que llama a Tdot.
    """
    def __init__(self, X: CSR):
        self.X = X

    def __matmul__(self, r: Array) -> Array:
        return self.X.Tdot(r)
