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


# -------------------------
# 4. Normalize
# -------------------------

faiss.normalize_L2(embeddings)


# -------------------------
# 5. Create FAISS index
# -------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -------------------------
# 6. User query
# -------------------------

query = "How can I work with data using Python?"

query_embedding = model.encode([query])

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)

faiss.normalize_L2(query_embedding)


# -------------------------
# 7. Search
# -------------------------

k = 3

scores, indices = index.search(
    query_embedding,
    k
)


# -------------------------
# 8. Retrieve chunks
# -------------------------

for score, index_number in zip(scores[0], indices[0]):

    chunk = chunks[index_number]

    print("=" * 60)
    print(f"Score: {score:.4f}")
    print(f"Source: {chunk['metadata']['source']}")
    print("Text:")
    print(chunk["text"])