#!/usr/bin/env bash
# run_ensemble.sh: ensemble multiple model predictions and create submission

set -euo pipefail  # 出錯就停在當行，未定義變數也當錯誤處理

########################################
# default config (all lower-case)
########################################
gpuid=0

lang=eng
domain=laptop

# 以逗號分隔的多個 prediction 資料夾，每個裡面要有 predictions.json
# 例如：
#   model_preddirs="exp/eng_laptop/baseline_outputs/test,exp/eng_laptop/modernbert_outputs/test"
model_preddirs=""

# 以逗號分隔的權重，可省略；省略時會自動平均
# 例如：
#   weights="0.3,0.7"
weights=""

# ensemble 後要輸出的資料夾
output_dir=./ensemble_outputs/eng_laptop_test

stage=1
stop_stage=100000

########################################
# env & options
########################################
. ./path.sh || exit 1
. ./parse_options.sh || exit 1

if [ -z "${model_preddirs}" ]; then
  echo "[run_ensemble.sh] --model_preddirs is required (comma-separated list)" >&2
  exit 1
fi

mkdir -p "${output_dir}"

# 指定要用哪張 GPU（其實 ensemble 不太用到 GPU，但為了一致性保留）
export CUDA_VISIBLE_DEVICES=${gpuid}
echo "[run_ensemble.sh] use gpu id: ${gpuid} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"

echo "[run_ensemble.sh] lang=${lang}, domain=${domain}"
echo "[run_ensemble.sh] model_preddirs=${model_preddirs}"
echo "[run_ensemble.sh] weights=${weights}"
echo "[run_ensemble.sh] output_dir=${output_dir}"
echo "[run_ensemble.sh] stage=${stage}, stop_stage=${stop_stage}"

########################################
# stage 1: ensemble predictions
########################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "===== stage 1: ensemble predictions ====="

  # 組 ensemble.py 指令
  cmd=(python ensemble.py
      --model_preddirs "${model_preddirs}"
      --output_dir "${output_dir}"
      --lang "${lang}"
      --domain "${domain}")

  if [ -n "${weights}" ]; then
    cmd+=(--weights "${weights}")
  fi

  echo "[run_ensemble.sh] running: ${cmd[*]}"
  "${cmd[@]}"
fi

########################################
# stage 2: create submission (using ensembled predictions)
########################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "===== stage 2: create submission from ensemble ====="
  # output_dir 裡應該已經有 pred_${lang}_${domain}.jsonl
  python create_submission.py \
    --output_dir "${output_dir}"
fi

echo "[run_ensemble.sh] done."
