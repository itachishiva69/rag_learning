from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------
# 1. Load documents
# -------------------------

data_dir = Path("data")

documents = []

for file_path in data_dir.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    documents.append({
        "text": text,
        "metadata": {
            "source": file_path.name
        }
    })


# -------------------------
# 2. Chunk documents
# -------------------------

chunk_size = 150
chunk_overlap = 30

chunks = []

for document in documents:
    text = document["text"]
    metadata = document["metadata"]

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "metadata": metadata.copy()
        })

        start += chunk_size - chunk_overlap


# -------------------------
# 3. Create embeddings
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(chunk_texts)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print("Chunks:", len(chunks))
print("Embeddings:", embeddings.shape)


# -------------------------
# 4. Normalize embeddings
# -------------------------

faiss.normalize_L2(embeddings)


# -------------------------
# 5. Create FAISS index
# -------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)


# -------------------------
# 6. Add embeddings
# -------------------------

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)