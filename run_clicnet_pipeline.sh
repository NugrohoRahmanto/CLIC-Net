#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p notebook/executed

# Editable run configuration.
#
# Leave SEED empty or set it to 0 to use the notebook default seed 42.
# Leave EPOCHS empty or set it to 0 to use the notebook default epoch count 50.
# Set the raw dataset roots here when auto-discovery is not desired.
# The processed data-preparation output remains:
#   dataset/ptb_xl
#   dataset/ptb_diagnostic
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
PTB_XL_RAW_ROOT="${PTB_XL_RAW_ROOT:-}"
PTB_DIAGNOSTIC_RAW_ROOT="${PTB_DIAGNOSTIC_RAW_ROOT:-}"

export CLICNET_SEED="$SEED"
export CLICNET_EPOCHS="$EPOCHS"
export CLICNET_PTB_XL_RAW_ROOT="$PTB_XL_RAW_ROOT"
export CLICNET_PTB_DIAGNOSTIC_RAW_ROOT="$PTB_DIAGNOSTIC_RAW_ROOT"

NOTEBOOKS=(
  "1_data_preparation.ipynb|1_data_preparation_executed.ipynb|DATA PREPARATION"
  "2_pretrain_clicnet.ipynb|2_pretrain_clicnet_executed.ipynb|PRETRAIN CLIC-NET"
  "3_finetune_clicnet.ipynb|3_finetune_clicnet_executed.ipynb|FINETUNE CLIC-NET"
  "4_pretrain_leadprior.ipynb|4_pretrain_leadprior_executed.ipynb|PRETRAIN LEAD PRIOR"
  "5_finetune_leadprior.ipynb|5_finetune_leadprior_executed.ipynb|FINETUNE LEAD PRIOR"
  "6_pretrain_cnntransformer.ipynb|6_pretrain_cnntransformer_executed.ipynb|PRETRAIN CNN-TRANSFORMER"
  "7_finetune_cnntransformer.ipynb|7_finetune_cnntransformer_executed.ipynb|FINETUNE CNN-TRANSFORMER"
  "8_pretrain_model_comparison.ipynb|8_pretrain_model_comparison_executed.ipynb|PRETRAIN MODEL COMPARISON"
  "9_finetune_model_comparison.ipynb|9_finetune_model_comparison_executed.ipynb|FINETUNE MODEL COMPARISON"
)

echo "====================================="
echo "CLIC-Net MI Localization Full Pipeline Started"
echo "Start Time: $(date)"
echo "Seed      : ${CLICNET_SEED:-42}"
echo "Epochs    : ${CLICNET_EPOCHS:-50}"
echo "PTB-XL raw root        : ${CLICNET_PTB_XL_RAW_ROOT:-auto-discover}"
echo "PTB Diagnostic raw root: ${CLICNET_PTB_DIAGNOSTIC_RAW_ROOT:-auto-discover}"
echo "====================================="

total="${#NOTEBOOKS[@]}"
step=0

for item in "${NOTEBOOKS[@]}"; do
  step=$((step + 1))
  IFS="|" read -r notebook_name executed_name label <<< "$item"
  notebook_path="notebook/${notebook_name}"

  if [[ ! -f "$notebook_path" ]]; then
    echo "Missing notebook: $notebook_path" >&2
    exit 1
  fi

  echo ""
  echo "[${step}/${total}] ${label} START"
  echo "Notebook: ${notebook_path}"
  echo "Time    : $(date)"
  echo ""

  .venv/bin/jupyter nbconvert \
    --to notebook \
    --execute "$notebook_path" \
    --ExecutePreprocessor.timeout=-1 \
    --output "$executed_name" \
    --output-dir notebook/executed

  echo ""
  echo "[${step}/${total}] ${label} FINISHED"
  echo "Time    : $(date)"
  echo ""
done

echo "====================================="
echo "CLIC-Net MI Localization Full Pipeline Completed"
echo "Finish Time: $(date)"
echo "====================================="
