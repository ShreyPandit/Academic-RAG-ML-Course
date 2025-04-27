import os
from .embedding_based import EmbeddingRetriver
from .bm25_based import BM25Retriver

INDEX_DIR = "./Data/index_store/"
CUDA_DEVICE = "cuda:0"

def build_retriever(type: str, topk: int, data_dir: str):
    if type == 'none':
        return None
    elif type == 'embed':
        embed_model_name = "BAAI/bge-small-en-v1.5"
        print(f"Using Embedding-based Retriever with {embed_model_name} model.")
        index_dir = os.path.join(INDEX_DIR, "embed_retreiver")
        return EmbeddingRetriver(data_dir, index_dir, embed_model_name, topk, CUDA_DEVICE)
    elif type == 'bm25':
        print(f"Using BM25-based Retriever.")
        index_dir = os.path.join(INDEX_DIR, "bm25_retreiver")
        return BM25Retriver(data_dir, index_dir, topk)
    else:
        print(f"Retreiver type {type} not known!")
        pass
