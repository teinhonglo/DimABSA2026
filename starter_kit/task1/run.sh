#!/usr/bin/env bash
# run.sh: pipeline for DimABSA track A subtask 1

set -euo pipefail

########################################
# default config (all lower-case)
########################################
gpuid=3

lang=eng
domain=laptop

data_root=./data
exp_root=./exp

conf=./conf/baseline.json

test_sets="dev test"
stage=1
stop_stage=100000
checkpoint=

########################################
# env & options
########################################
. ./path.sh || exit 1
. ./parse_options.sh || exit 1

data_dir=${data_root}/${lang}_${domain}
exp_tag=$(basename "${conf}" .json)
exp_dir=${exp_root}/${exp_tag}/${lang}_${domain}
out_dir=${exp_dir}/

mkdir -p "${data_dir}" "${exp_dir}" "${out_dir}"

# 指定要用哪張 GPU
export CUDA_VISIBLE_DEVICES=${gpuid}
echo "[run.sh] use gpu id: ${gpuid} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"

echo "[run.sh] lang=${lang}, domain=${domain}"
echo "[run.sh] data_dir=${data_dir}"
echo "[run.sh] exp_dir=${exp_dir}"
echo "[run.sh] out_dir=${out_dir}"
echo "[run.sh] conf=${conf}"
echo "[run.sh] stage=${stage}, stop_stage=${stop_stage}"

########################################
# stage 1: data preparation
########################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "===== stage 1: data preparation ====="
  python data_prep.py \
    --lang "${lang}" \
    --domain "${domain}" \
    --out_data_dir "${data_dir}"
fi

########################################
# stage 2: train
########################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "===== stage 2: train ====="

  train_extra_args=()
  if [ -n "${checkpoint}" ]; then
    echo "[run.sh] use checkpoint: ${checkpoint}"
    train_extra_args+=(--checkpoint "${checkpoint}")
  fi
  python train.py \
    --train_json "${data_dir}/train.json" \
    --dev_json   "${data_dir}/dev.json" \
    --model_conf "${conf}" \
    --exp_dir    "${exp_dir}" \
    "${train_extra_args[@]}"
fi

########################################
# stage 3: test / inference
########################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "===== stage 3: test / inference ====="
  for test_set in $test_sets; do
    python test.py \
      --test_json  "${data_dir}/${test_set}.json" \
      --model_dir  "${exp_dir}" \
      --output_dir "${out_dir}"/$test_set
   done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "===== stage 4: evaluation: dev set ====="
  python evaluation.py \
    --gold_json "${data_dir}/dev.json" \
    --pred_json "${out_dir}/dev/predictions.json" > ${out_dir}/dev/metrics.log
    head ${out_dir}/dev/metrics.log
fi


########################################
# stage 5: create submission (test set)
########################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "===== stage 5: create submission: test set ====="
  mkdir -p ${exp_root}/${exp_tag}/submission
  cp ${out_dir}/test/pred_* ${exp_root}/${exp_tag}/submission/
  python create_submission.py \
    --output_dir "${exp_root}/${exp_tag}/submission"
fi

echo "[run.sh] done."
