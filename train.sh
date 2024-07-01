#!/bin/bash
BASE_MODEL_DIR="PATH_OF_BASE_MODEL"
TRAIN_DATA_LIST="TRAIN_DATASETS"
POS_DIR="PATH_TO_POS_LOGITS"
NEG_DIR="PATH_TO_NEG_LOGITS"
DATA_DIR="DATASET_DIR"
INBATCH_PKL_PATH_DIR="PATH_TO_INBATCH_LOGITS_PKL"
FEATURE_PKL_PATH_DIR="PATH_TO_FEATURE_PKL"
BATCH_SIZE=32
NEG_K=8
NUM_HEADS=32
HIDDEN_DIM=512
OUTPUT_DIM=1
LN="True"
NORM="False"
PADDING_SIDE="right"
NUM_EPOCHS=5
MAX_SEQ_LENGTH=250
LR=1e-4
ALPHA=1
BETA=1
GAMMA=0.01
ETA=0.001
TEMPERATURE_IN_BATCH=1
TEMPERATURE_HARDNEG=1
TEMPERATURE_TEACHER_HARDNEG=1
SCALE_PARAM=1
LOG_INTERVAL=10
EVAL_INTERVAL=300
TB_DIR="PATH_TO_TENSORBOARD_PATH"
PATIENCE=5
NUM_CKPT=4
TRAINING_LOG="PATH_TO_TRAINING_LOG"
OUTPUT_DIR="PATH_TO_OUTPUT_MODEL"

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12346}

python -m torch.distributed.run --nproc_per_node=$gpus --nnode=$WORLD_SIZE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                                train.py --base_model_dir $BASE_MODEL_DIR \
                                --train_data_list $TRAIN_DATA_LIST \
                                --pos_dir $POS_DIR \
                                --neg_dir $NEG_DIR \
                                --data_dir $DATA_DIR \
                                --inbatch_pkl_path_dir $INBATCH_PKL_PATH_DIR \
                                --feature_pkl_path_dir $FEATURE_PKL_PATH_DIR \
                                --batch_size $BATCH_SIZE
                                --neg_K $NEG_K \
                                --num_heads $NUM_HEADS \
                                --hidden_dim $HIDDEN_DIM \
                                --output_dim $OUTPUT_DIM \
                                --ln $LN \
                                --norm $NORM \
                                --num_epochs $NUM_EPOCHS \
                                --padding_side $PADDING_SIDE \
                                --max_seq_length $MAX_SEQ_LENGTH \
                                --lr $LR \
                                --alpha $ALPHA \
                                --beta $BETA \
                                --gamma $GAMMA \
                                --eta $ETA \
                                --temperature_in_batch $TEMPERATURE_IN_BATCH \
                                --temperature_hardneg $TEMPERATURE_HARDNEG \
                                --temperature_teacher_hardneg $TEMPERATURE_TEACHER_HARDNEG \
                                --scale_param $SCALE_PARAM \
                                --log_interval $LOG_INTERVAL \
                                --eval_interval $EVAL_INTERVAL \
                                --tb_dir $TB_DIR \
                                --patience $PATIENCE \
                                --num_ckpt $NUM_CKPT \
                                --training_log $TRAINING_LOG \
                                --output_dir $OUTPUT_DIR \