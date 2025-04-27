import os
from .embedding_based import EmbeddingRetriver

INDEX_DIR = "./Data/index_store/"
CUDA_DEVICE = "cuda:0"

def build_retriever(type: str, topk: int, data_dir: str):
    if type == 'none':
        return None
    elif type == 'embed':
        index_dir = os.path.join(INDEX_DIR, "embed_retreiver")
        embed_model_name = "BAAI/bge-small-en-v1.5"
        return EmbeddingRetriver(data_dir, index_dir, embed_model_name, topk, CUDA_DEVICE)
    elif type == 'bm25':
        index_dir = os.path.join(INDEX_DIR, "bm25_retreiver")
    else:
        print(f"Retreiver type {type} not known!")
        pass
