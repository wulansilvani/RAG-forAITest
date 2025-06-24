import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


FAISS_INDEX_PATH = Path("data/faiss_index/index.faiss")
ID_MAP_PATH = Path("data/faiss_index/id_map.pkl")
CHUNK_DIR = Path("data/chunks")
TOP_K = 5  

# Load FAISS index and ID map
index = faiss.read_index(str(FAISS_INDEX_PATH))
with open(ID_MAP_PATH, "rb") as f:
    id_map = pickle.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query: str, top_k=TOP_K):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    results = []

    for i in indices[0]:
        if i < len(id_map):
            chunk_file = CHUNK_DIR / id_map[i]
            if chunk_file.exists():
                results.append(chunk_file.read_text(encoding="utf-8"))
    return results

if __name__ == "__main__":
    question = "What are the common techniques for phishing?"
    chunks = retrieve_chunks(question)
    print("Retrieved Chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---\n{chunk[:500]}\n")
