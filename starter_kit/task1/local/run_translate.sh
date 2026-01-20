#!/usr/bin/env bash
# run.sh: pipeline for DimABSA track A subtask 1

set -euo pipefail
OPENAI_API_KEY=$OPENAI_API_KEY

python local/translate_mlang2eng_multi.py \
  --data_root ./data \
  --out_dir ./data/mlang2eng_multi \
  --model gpt-4.1
