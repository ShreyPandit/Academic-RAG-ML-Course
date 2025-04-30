import json
import glob
import os
from llama_index.core import VectorStoreIndex, Document, load_index_from_storage, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.postprocessor import SentenceTransformerRerank

# def load_chunks(json_dir):
#     docs = []
#     for json_path in glob.glob(os.path.join(json_dir, "*.json")):
#         with open(json_path, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         docs.extend(
#             Document(text=entry["chunk"], doc_id=entry.get("id"))
#             for entry in raw
#         )
#     return docs

# embedder = HuggingFaceEmbedding(
#     model_name=EMBED_MODEL_NAME,
#     device=DEVICE,
#     cache_folder=CACHE_DIR,
# )

# if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
#     storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
#     index = load_index_from_storage(
#         storage_context,
#         embed_model=embedder
#     )
# else:
#     documents = load_chunks(JSON_CHUNKS_DIR)
#     index = VectorStoreIndex.from_documents(
#         documents,
#         embed_model=embedder,
#     )
#     index.storage_context.persist(persist_dir=INDEX_DIR)


# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=TOP_K,
#     embed_model=embedder,
# )

# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=TOP_K,
#     embed_model=embedder,
# )

# def retrieve(question: str):
#     if not question.strip():
#         return []
#     hits = retriever.retrieve(question)
#     return [doc.text for doc in hits]


class EmbeddingRetriver:
    def __init__(self, data_path, index_dir, embed_model_name, topk, use_reranker, device):
        self.embed_model_name = embed_model_name
        self.embedder = HuggingFaceEmbedding(
            model_name=self.embed_model_name,
            device=device,
        )
        self.topk = topk
        self.data_path = data_path
        self.index_dir = index_dir
        self.index = self.build_index(self.index_dir)
        self.retriever = self.build_retriever()
        if use_reranker:
            print(f"Using reranker with top n = topk!")
            self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
            top_n=self.topk
            )
        else:
            self.reranker = None
        
    def load_chunks(self, path):
        docs = []
        for json_path in glob.glob(os.path.join(path, "*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            docs.extend(
                Document(text=entry["chunk"], doc_id=entry.get("id"))
                for entry in raw
            )
        return docs

    def build_index(self, index_dir):
        if os.path.isdir(index_dir) and os.listdir(index_dir):
            print("Loading existing index from disk...")
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            index = load_index_from_storage(
                storage_context,
                embed_model=self.embedder
            )
        else:
            print("No existing index found. Building index...")
            documents = self.load_chunks(self.data_path)
            print(f"Loaded {len(documents)} non-empty documents.")
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embedder,
            )
            index.storage_context.persist(persist_dir=index_dir)
        return index
        
    def build_retriever(self):
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.topk,
            embed_model=self.embedder,
        )
        return retriever

    def retrieve(self, text: str):
        if not text.strip():
            return []
        hits = self.retriever.retrieve(text)
        if self.reranker is not None:
            hits = self.rerank(hits, text)
        return [doc.text for doc in hits]

    def rerank(self, retrieved_nodes, query):
        query_bundle = QueryBundle(query)
        retrieved_nodes = self.reranker.postprocess_nodes(retrieved_nodes, query_bundle)
        return retrieved_nodes


if __name__ == "__main__":
    JSON_CHUNKS_DIR = "Data"
    INDEX_DIR = "./Data/index_store/"
    # EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    EMBED_MODEL_NAME = "Salesforce/SFR-Embedding-2_R"
    # EMBED_MODEL_NAME = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
    # EMBED_MODEL_NAME = "witiko/mathberta"
    # CACHE_DIR       = "cache"
    DEVICE           = "cuda:0"
    TOP_K            = 5
    RERANKER  = True
    q = """ 
    For which matrices does the singular-value decomposition (SVD) exist?
    """
    INDEX_DIR = os.path.join(INDEX_DIR, "embed_retriever", EMBED_MODEL_NAME.split('/')[-1].replace('-', '_').replace('.', '_'))
    retriever = EmbeddingRetriver(JSON_CHUNKS_DIR, INDEX_DIR, EMBED_MODEL_NAME, TOP_K, RERANKER, DEVICE)
    for i, chunk in enumerate(retriever.retrieve(q), 1):
        print(f"\n=== Passage #{i} ===\n{chunk}")
