import json
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

JSON_CHUNKS_PATH = "Academic-RAG-ML-Course/Retriever/textbook_chunks_temp.json"
CACHE_DIR         = "Academic-RAG-ML-Course/cache"
DEVICE            = "cuda:0"
EMBED_MODEL_NAME  = "BAAI/bge-small-en-v1.5"
TOP_K             = 5

def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs = [
        Document(text=entry["chunk"], doc_id=entry.get("id"))
        for entry in raw
    ]
    return docs

embedder = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_NAME,
    device=DEVICE,
    cache_folder=CACHE_DIR,
)

documents = load_chunks(JSON_CHUNKS_PATH)


index = VectorStoreIndex.from_documents(
    documents,
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
    top_chunks = retrieve(q)
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n=== Passage #{i} ===\n{chunk}")
