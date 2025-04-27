import json
import glob
import os
from llama_index.core import VectorStoreIndex, Document, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext

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
    def __init__(self, data_path, index_dir, embed_model_name, topk, device):
        self.embed_model_name = embed_model_name
        self.embedder = embedder = HuggingFaceEmbedding(
            model_name=self.embed_model_name,
            device=device,
        )
        self.topk = topk
        self.data_path = data_path
        self.index_dir = index_dir
        self.index = self.build_index(self.index_dir)
        self.retriever = self.build_retriever()
        
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
        return [doc.text for doc in hits]


if __name__ == "__main__":
    JSON_CHUNKS_DIR = "Data"
    INDEX_DIR       = "Data/index_storage"
    CACHE_DIR       = "cache"
    DEVICE           = "cuda:0"
    EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
    TOP_K            = 5
    q = """ 
    What is the derivative of the sigmoid function?
    """
    retriever = EmbeddingRetriver(JSON_CHUNKS_DIR, INDEX_DIR, EMBED_MODEL_NAME, TOP_K, DEVICE)
    for i, chunk in enumerate(retriever.retrieve(q), 1):
        print(f"\n=== Passage #{i} ===\n{chunk}")
