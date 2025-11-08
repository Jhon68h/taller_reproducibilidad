# io_libsvm.py
# Carga de datos en formato LIBSVM.
# - Local: devuelve (CSR, y, n_features)
# - Spark: devuelve (RDD[(idx_np, val_np, y_float)], n_features)
#
# Notas:
# * Convierte índices de 1-based (archivo) a 0-based (interno).
# * Etiquetas se mapean a {+1.0, -1.0}: label > 0 -> +1.0 ; en caso contrario -> -1.0.
# * Si bias >= 0, añade una columna virtual al final con valor 'bias' en todas las filas.

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

from csr import CSR


def _parse_libsvm_line(line: str) -> Tuple[float, List[int], List[float]]:
    """
    Parsea una línea LIBSVM -> (y_float, idx_list_0based, val_list)
    Ignora comentarios con '#'.
    """
    # Quitar comentario y espacios
    line = line.split('#', 1)[0].strip()
    if not line:
        raise ValueError("empty")

    parts = line.split()
    y_raw = parts[0]
    try:
        y_val = float(y_raw)
    except Exception:
        # En casos raros (e.g., etiquetas no numéricas), forzamos -1
        y_val = -1.0
    y = 1.0 if y_val > 0 else -1.0

    idxs: List[int] = []
    vals: List[float] = []
    for token in parts[1:]:
        if ':' not in token:
            # tokens extra (e.g., qid) se ignoran
            continue
        k, v = token.split(':', 1)
        if not k:
            continue
        try:
            j = int(k) - 1  # 1-based -> 0-based
            if j < 0:
                continue
            x = float(v)
        except Exception:
            continue
        idxs.append(j)
        vals.append(x)

    # Opcional: combinar índices repetidos en la misma fila (poco común)
    if len(idxs) >= 2:
        # ordenar por índice y sumar duplicados
        order = np.argsort(idxs)
        idxs_sorted = [idxs[i] for i in order]
        vals_sorted = [vals[i] for i in order]
        uniq_idx: List[int] = []
        uniq_val: List[float] = []
        last = None
        acc = 0.0
        for j, x in zip(idxs_sorted, vals_sorted):
            if last is None:
                last = j
                acc = x
            elif j == last:
                acc += x
            else:
                uniq_idx.append(last)
                uniq_val.append(acc)
                last = j
                acc = x
        if last is not None:
            uniq_idx.append(last)
            uniq_val.append(acc)
        idxs, vals = uniq_idx, uniq_val

    return y, idxs, vals


def load_libsvm_local(path: str, bias: float = -1.0) -> Tuple[CSR, np.ndarray, int]:
    """
    Carga local del archivo LIBSVM.
    Devuelve:
      X: CSR (csr.CSR)
      y: np.ndarray (float64, valores en {+1.0, -1.0})
      n_features: columnas totales (incluido bias si aplica)
    """
    y_list: List[float] = []
    rows_idx: List[List[int]] = []
    rows_val: List[List[float]] = []
    max_j = -1

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                y, idxs, vals = _parse_libsvm_line(line)
            except ValueError:
                # línea vacía tras limpiar
                continue

            if idxs:
                mj = max(idxs)
                if mj > max_j:
                    max_j = mj

            y_list.append(y)
            rows_idx.append(idxs)
            rows_val.append(vals)

    n_rows = len(y_list)
    if n_rows == 0:
        raise RuntimeError(f"No se encontraron muestras válidas en: {path}")

    n_features = max_j + 1
    add_bias = bias is not None and bias >= 0.0
    if add_bias:
        bias_col = n_features  # nueva última columna
        n_features += 1

    # Construir CSR
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    nnz = 0
    for i in range(n_rows):
        nnz_i = len(rows_idx[i])
        if add_bias:
            nnz_i += 1
        nnz += nnz_i
        indptr[i + 1] = nnz

    indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)

    cursor = 0
    for i in range(n_rows):
        idxs = rows_idx[i]
        vals = rows_val[i]
        # Copiar pares existentes
        if idxs:
            li = len(idxs)
            indices[cursor:cursor + li] = np.asarray(idxs, dtype=np.int64)
            data[cursor:cursor + li] = np.asarray(vals, dtype=np.float64)
            cursor += li
        # Añadir bias si corresponde
        if add_bias:
            indices[cursor] = bias_col # type: ignore
            data[cursor] = float(bias)
            cursor += 1

    X = CSR(indptr=indptr, indices=indices, data=data, n_cols=n_features)
    y = np.asarray(y_list, dtype=np.float64)

    return X, y, n_features


# ---------------------- Carga con PySpark (opcional) ---------------------- #

def _parse_partition(iter_lines, bias: Optional[float]):
    """
    parsea líneas en una partición -> genera tuplas (idx_np, val_np, y_float), max_idx_local
    Se emiten dos tipos de registros:
      ('data', (idx_np, val_np, y_float))
      ('maxj', int_max_indice_en_particion)
    """
    max_j_local = -1
    out = []
    for line in iter_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            y, idxs, vals = _parse_libsvm_line(line)
        except ValueError:
            continue

        if idxs:
            mj = max(idxs)
            if mj > max_j_local:
                max_j_local = mj

        idx_np = np.asarray(idxs, dtype=np.int64)
        val_np = np.asarray(vals, dtype=np.float64)
        out.append(('data', (idx_np, val_np, float(y))))

    # Emitir primero todos los datos, luego un registro con el max local
    for rec in out:
        yield rec
    yield ('maxj', max_j_local)


def load_libsvm_rdd(sc, path: str, numPartitions: Optional[int] = None, bias: float = -1.0):
    """
    Carga distribuida del archivo LIBSVM vía PySpark.
    Devuelve:
      rdd: RDD de (idx_np[int64], val_np[float64], y_float)
      n_features: int (incluye columna de bias si aplica)
    """
    if numPartitions is None:
        rdd_raw = sc.textFile(path)
    else:
        rdd_raw = sc.textFile(path, minPartitions=int(numPartitions))

    # Parse por partición, capturando max_j local
    parsed = rdd_raw.mapPartitions(lambda it: _parse_partition(it, bias))

    # Separar datos y maximos
    data_rdd = parsed.filter(lambda kv: kv[0] == 'data').map(lambda kv: kv[1])
    maxj_rdd = parsed.filter(lambda kv: kv[0] == 'maxj').map(lambda kv: kv[1])

    if maxj_rdd.isEmpty():
        raise RuntimeError("No se encontraron datos válidos en el archivo.")

    max_j = maxj_rdd.max()
    n_features = int(max_j + 1)

    add_bias = bias is not None and bias >= 0.0
    if add_bias:
        bias_col = n_features
        n_features += 1
        bc_bias_val = sc.broadcast(float(bias))
        bc_bias_col = sc.broadcast(int(bias_col))

        def _append_bias(rec):
            idx, val, y = rec
            # Añadir una entrada más al final (no es necesario ordenar)
            idx2 = np.empty(idx.size + 1, dtype=np.int64)
            val2 = np.empty(val.size + 1, dtype=np.float64)
            idx2[:-1] = idx
            val2[:-1] = val
            idx2[-1] = bc_bias_col.value
            val2[-1] = bc_bias_val.value
            return (idx2, val2, y)

        data_rdd = data_rdd.map(_append_bias)

    return data_rdd, n_features
