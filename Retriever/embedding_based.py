import json
import glob
import os
from llama_index.core import VectorStoreIndex, Document, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext

JSON_CHUNKS_DIR = "Academic-RAG-ML-Course/Data"
INDEX_DIR       = "Academic-RAG-ML-Course/Data/index_storage"
CACHE_DIR       = "Academic-RAG-ML-Course/cache"

DEVICE           = "cuda:0"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K            = 5

def load_chunks(json_dir):
    docs = []
    for json_path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        docs.extend(
            Document(text=entry["chunk"], doc_id=entry.get("id"))
            for entry in raw
        )
    return docs

embedder = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    device=DEVICE,
    cache_folder=CACHE_DIR,
)

if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(
        storage_context,
        embed_model=embedder
    )
else:
    documents = load_chunks(JSON_CHUNKS_DIR)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embedder,
    )
    index.storage_context.persist(persist_dir=INDEX_DIR)


retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=TOP_K,
    embed_model=embedder,
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=TOP_K,
    embed_model=embedder,
)

def retrieve(question: str):
    if not question.strip():
        return []
    hits = retriever.retrieve(question)
    return [doc.text for doc in hits]

if __name__ == "__main__":
    q = "What is the definition of photosynthesis?"
    for i, chunk in enumerate(retrieve(q), 1):
        print(f"\n=== Passage #{i} ===\n{chunk}")
