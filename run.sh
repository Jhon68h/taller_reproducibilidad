#!/usr/bin/env bash
set -euo pipefail

# ------------------------- Config base ------------------------- #
DATA="/home/jhonatan/Documents/analisis_de_datos_a_gran_escala/taller_reproducibilidad/dataset/webspam_wc_normalized_unigram.svm"
EPS="1e-3"
MAXITER="100"
SEED_MAIN="42"
TRAIN_RATIO="0.8"     # split inicial train/test
VAL_RATIO="0.8"       # porcentaje de train que se queda como train (resto es val)
CVALS=("0.1" "1.0" "10.0")
REPS=("1" "2" "3")

command -v python >/dev/null 2>&1 || { echo "ERROR: python no está en PATH"; exit 1; }
[[ -f "$DATA" ]] || { echo "ERROR: no existe el archivo: $DATA" >&2; exit 2; }

# --------------------- Carpetas por corrida -------------------- #
RUN_TAG=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/${RUN_TAG}"
MODELDIR="models/${RUN_TAG}"
OUTDIR="graphics/${RUN_TAG}"
mkdir -p data "${LOGDIR}" "${MODELDIR}" "${OUTDIR}"

# ------------------------- Split train/test ------------------------- #
TRAIN="data/train.svm"
TEST="data/test.svm"

echo "[1/6] Split dataset -> $TRAIN / $TEST"
python src/split_libsvm.py \
  --data "$DATA" \
  --train_out "$TRAIN" \
  --test_out "$TEST" \
  --train_ratio "$TRAIN_RATIO" \
  --seed "$SEED_MAIN" | tee "${LOGDIR}/00_split_train_test.log"

# ------------------- Split interno train/val ------------------- #
TRAIN_ONLY="data/train_only.svm"
VAL="data/val.svm"
echo "[2/6] Split interno de train -> $TRAIN_ONLY / $VAL"
python src/split_libsvm.py \
  --data "$TRAIN" \
  --train_out "$TRAIN_ONLY" \
  --test_out "$VAL" \
  --train_ratio "$VAL_RATIO" \
  --seed 7 | tee "${LOGDIR}/01_split_train_val.log"

# Sustituye train por train_only (dejando VAL independiente)
mv -f "$TRAIN_ONLY" "$TRAIN"

# ------------------- Función: correr experimento ------------------- #
run_exp () {
  local SOLVER="$1"    # lr | l2svm
  local CVAL="$2"
  local BIAS="$3"      # -1 o 1.0
  local REP="$4"       # repetición (seed)
  local TAG="${SOLVER}_C${CVAL}_eps${EPS}_bias${BIAS}_rep${REP}"

  echo "[*] Entrenando ${TAG}"
  python src/train.py \
    --mode local \
    --data "$TRAIN" \
    --solver "$SOLVER" \
    --C "$CVAL" \
    --eps "$EPS" \
    --bias "$BIAS" \
    --maxIter "$MAXITER" \
    --seed "$REP" \
    --log_json "${LOGDIR}/metrics_${TAG}.json" | tee "${LOGDIR}/10_train_${TAG}.log"

  [[ -f models/w.npy ]] || { echo "ERROR: models/w.npy no existe tras entrenamiento (${TAG})"; exit 3; }
  cp models/w.npy "${MODELDIR}/w_${TAG}.npy"

  # Si es LR: buscar umbral óptimo en validación
  local THR_ARG=()
  if [[ "$SOLVER" == "lr" ]]; then
    python src/find_threshold.py \
      --data "$VAL" \
      --weights "${MODELDIR}/w_${TAG}.npy" \
      --bias "$BIAS" \
      --metric acc \
      --out "${LOGDIR}/threshold_${TAG}.json" | tee "${LOGDIR}/12_thr_${TAG}.log"

    # EXTRAER best_threshold del JSON (sin jq), usando -c
    BEST_THR=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_threshold"])' \
               "${LOGDIR}/threshold_${TAG}.json")

    THR_ARG=(--threshold "$BEST_THR")
  fi


  echo "[*] Evaluando test: ${TAG}"
  python src/eval.py \
    --data "$TEST" \
    --weights "${MODELDIR}/w_${TAG}.npy" \
    --bias "$BIAS" \
    "${THR_ARG[@]}" | tee "${LOGDIR}/20_eval_${TAG}.log"
}

# ------------------- Barrido de C y repeticiones ------------------- #
echo "[3/6] Experimentos LR (sin bias) con ajuste de umbral en validación"
for C in "${CVALS[@]}"; do
  for R in "${REPS[@]}"; do
    run_exp "lr" "$C" "-1" "$R"
  done
done

echo "[4/6] Experimentos L2-SVM (sin bias)"
for C in "${CVALS[@]}"; do
  for R in "${REPS[@]}"; do
    run_exp "l2svm" "$C" "-1" "$R"
  done
done

# ------------------- Gráficas para esta corrida ------------------- #
echo "[5/6] Generando gráficas en ${OUTDIR}"
python src/plot_metrics.py \
  --logs_glob "${LOGDIR}/metrics_*.json" \
  --outdir "${OUTDIR}" \
  --test "$TEST"

# ------------------- Resumen final ------------------- #
echo "[6/6] Resumen:"
echo "  - Logs:     ${LOGDIR}"
echo "  - Modelos:  ${MODELDIR}"
echo "  - Gráficas: ${OUTDIR}"
echo "  - JSONs de métricas: ${LOGDIR}/metrics_*.json"
