#!/usr/bin/env bash
set -euo pipefail

DATA="/home/jhonatan/Documents/analisis_de_datos_a_gran_escala/taller_reproducibilidad/dataset/webspam_wc_normalized_unigram.svm"

# Opción 2: pásala como primer argumento al ejecutar:
#   bash run_local.sh /ruta/a/mi_dataset.svm
if [[ $# -ge 1 ]]; then
  DATA="$1"
fi

# Hiperparámetros básicos
SOLVER="lr"         # "lr" (logística) o "l2svm" (hinge^2)
CVAL="1.0"          # regularización
EPS="1e-2"          # tolerancia TRON
BIAS="-1"           # -1 sin bias; usa 1.0 para añadir columna de sesgo
MAXITER="100"       # iteraciones máximas

# Ejecuta el entrenamiento en modo local
python src/train.py \
  --mode local \
  --data "${DATA}" \
  --solver "${SOLVER}" \
  --C "${CVAL}" \
  --eps "${EPS}" \
  --bias "${BIAS}" \
  --maxIter "${MAXITER}"
