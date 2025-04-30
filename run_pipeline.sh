#! /bin/bash

DATA_DIR="./Data/"
EVAL_DATA="./questions.json"
RETRIEVER="embed:BAAI/bge-large-en-v1.5"
# RETRIEVER="embed:Salesforce/SFR-Embedding-2_R"
# RETRIEVER="embed:math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
TOPK=$1
RERANK_ENABLED=1
echo "Running with setting: ${RETRIEVER} TOPK: ${TOPK} RERANK: ${RERANK_ENABLED}"

if [ "$RERANK_ENABLED" -eq 1 ]; then
    RUN_NAME="${RETRIEVER}_topk${TOPK}_rerank.json"
    SAVE_FILE="./saves/${RUN_NAME}"
    python pipeline.py --data_dir $DATA_DIR \
        --eval_data $EVAL_DATA \
        --retriever $RETRIEVER \
        --topk $TOPK \
        --save $SAVE_FILE \
        --use_reranker
else
    RUN_NAME="${RETRIEVER}_topk${TOPK}.json"
    SAVE_FILE="./saves/${RUN_NAME}"
    python pipeline.py --data_dir $DATA_DIR \
        --eval_data $EVAL_DATA \
        --retriever $RETRIEVER \
        --topk $TOPK \
        --save $SAVE_FILE
fi
