import json
import glob
import os
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.schema import TextNode
# from llama_index.core.storage.storage_context import StorageContext

import json

# with open("textbook_chunks_temp.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# print(f"Number of entries: {len(data)}")
# for i, entry in enumerate(data[:5]): 
#     print(f"Entry #{i+1}: {entry}")

# def load_chunks(json_dir):
#     docs = []
#     for json_path in glob.glob(os.path.join(json_dir, "*.json")):
#         with open(json_path, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         for entry in raw:
#             chunk_text = entry.get("chunk", "").strip()
#             if chunk_text:  
#                 docs.append(TextNode(text=chunk_text, id_=entry.get("id")))
#     return docs

# if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
#     print("Loading existing BM25 index from disk...")
#     storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
#     docstore = SimpleDocumentStore.from_persist_path(INDEX_DIR)
# else:
#     print("No existing index found. Building BM25 index...")
#     documents = load_chunks(JSON_CHUNKS_DIR)
#     print(f"Loaded {len(documents)} non-empty documents.")
    
#     docstore = SimpleDocumentStore()
#     docstore.add_documents(documents) 
    
#     os.makedirs(INDEX_DIR, exist_ok=True)
#     docstore.persist()

# print(f"Number of documents in docstore: {len(docstore.docs)}")
# for node_id, node in docstore.docs.items():
#     print(f"Doc ID: {node_id}, Text Length: {len(node.text)}")

# bm25_retriever = BM25Retriever.from_defaults(
#     docstore=docstore,
#     similarity_top_k=TOP_K,
# )

# Re-ranker (R3)
# def keyword_reranker(query, nodes):
#     ranked = sorted(
#         nodes,
#         key=lambda node: sum(1 for word in query.split() if word.lower() in node.text.lower()),
#         reverse=True
#     )
#     return ranked

# Retriever
# def retrieve(question: str):
#     if not question.strip():
#         return []
#     hits = bm25_retriever.retrieve(question)  # retrieve from BM25
#     reranked_hits = keyword_reranker(question, hits)  # rerank using keyword overlap
#     return [doc.text for doc in reranked_hits]  # return only text


class BM25Retriver:
    def __init__(self, data_path, index_dir, topk):
        self.topk = topk
        self.data_path = data_path
        self.index_dir = index_dir
        self.docstore = self.build_index(self.index_dir)
        self.retriever = self.build_retriever()

    def load_chunks(self, path):
        docs = []
        for json_path in glob.glob(os.path.join(path, "*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw:
                chunk_text = entry.get("chunk", "").strip()
                if chunk_text:  
                    docs.append(TextNode(text=chunk_text, id_=entry.get("id")))
        return docs

    def build_index(self, index_dir):
        if os.path.isdir(index_dir) and os.listdir(index_dir):
            print("Loading existing BM25 index from disk...")
            # storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            docstore = SimpleDocumentStore.from_persist_path(os.path.join(index_dir, 'docstore.json'))
        else:
            print("No existing index found. Building BM25 index...")
            documents = self.load_chunks(self.data_path)
            print(f"Loaded {len(documents)} non-empty documents.")
            docstore = SimpleDocumentStore()
            docstore.add_documents(documents) 
            
            os.makedirs(index_dir, exist_ok=True)
            docstore.persist(persist_path=os.path.join(index_dir, 'docstore.json'))
        return docstore

    def build_retriever(self):
        retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,
            similarity_top_k=self.topk,
        )
        return retriever

    def retrieve(self, text: str):
        if not text.strip():
            return []
        hits = self.retriever.retrieve(text)  # retrieve from BM25
        # hits = keyword_reranker(text, hits)  # rerank using keyword overlap
        return [doc.text for doc in hits]  # return only text
    
if __name__ == "__main__":
    JSON_CHUNKS_DIR = "Data"
    INDEX_DIR       = "./Data/index_store/bm25_retriever/"
    TOP_K           = 5
    q = """ 
    What is the derivative of the sigmoid function?
    """
    print(f"\n>>> Query: {q}")
    retriever = BM25Retriver(JSON_CHUNKS_DIR, INDEX_DIR, TOP_K)
    for i, chunk in enumerate(retriever.retrieve(q), 1):
        print(f"\n=== Passage #{i} ===\n{chunk}")