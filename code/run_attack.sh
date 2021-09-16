#!/bin/bash

set -e
INPUT_DIR='../data/dev_images'
OUTPUT_DIR='./outputs'
MAX_EPSILON=16
EXP_NAME='exp_test'
CHECKPOINT_PATH='../model'
echo $EXP_NAME

python attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --max_epsilon="${MAX_EPSILON}" \
  --num_iter=10 \
  --momentum=1.0 \
  --prob=0.7 \
  --exp_name="${EXP_NAME}"\
  --batch_size=5

python simple_eval.py \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --exp_name="${EXP_NAME}"

