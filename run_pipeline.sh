#! /bin/bash

DATA_DIR="./Data/"
EVAL_DATA="./questions.json"
RETRIEVER=$1
TOPK=$2
RUN_NAME="${RETRIEVER}_topk${TOPK}.json"
SAVE_FILE="./saves/${RUN_NAME}"
python pipeline.py --data_dir $DATA_DIR \
    --eval_data $EVAL_DATA \
    --retriever $RETRIEVER \
    --topk $TOPK \
    --save $SAVE_FILE 
    