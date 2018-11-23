#!/bin/sh

USR_DIR=ner
PROBLEM=ner_nsml90k_subword_v2
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=/data/t2t_data/$PROBLEM
TMP_DIR=/data/t2t_datagen
TRAIN_DIR=/data/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
# * The data should be stored in a file named `ner_nsml90k_train_data` in $TMP_DIR
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
WORKER_GPU=1  # The number of GPUs for training
export CUDA_VISIBLE_DEVICES=0  # GPU IDs to be used

t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --worker_gpu=$WORKER_GPU \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=5000'

# Decode
DECODE_TO_FILE=$TRAIN_DIR/decode/test  # 디코딩 출력파일 이름의 prefix

BEAM_SIZE=4
ALPHA=0.6

export CUDA_VISIBLE_DEVICES=0  # GPU IDs to be used

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --eval_use_test_set=True \
  --decode_to_file=$DECODE_TO_FILE
