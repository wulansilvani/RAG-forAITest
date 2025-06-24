import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

CHUNK_DIR = Path("data/chunks")
FAISS_INDEX_PATH = Path("data/faiss_index/index.faiss")
ID_MAP_PATH = Path("data/faiss_index/id_map.pkl")
FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and accurate
chunk_files = sorted(CHUNK_DIR.glob("*.txt"))
documents = [f.read_text(encoding="utf-8") for f in chunk_files]
embeddings = model.encode(documents)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, str(FAISS_INDEX_PATH))    # Save
with open(ID_MAP_PATH, "wb") as f:
    pickle.dump([str(f.name) for f in chunk_files], f)

print(f"Embedded {len(documents)} chunks and saved FAISS index.")
