#! /bin/bash

RUN_NAME=$1
DATA_DIR="./Data/"
EVAL_DATA="./questions.json"
RETRIEVER="bm25"
TOPK=5
SAVE_FILE="./saves/${RUN_NAME}"
CUDA_VISIBLE_DEVICES=3 python pipeline.py --data_dir $DATA_DIR \
    --eval_data $EVAL_DATA \
    --retriever $RETRIEVER \
    --topk $TOPK \
    --save $SAVE_FILE 
    