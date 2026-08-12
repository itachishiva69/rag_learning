from pathlib import Path
from sentence_transformers import SentenceTransformer

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


# Chunking
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


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract only the text from each chunk
chunk_texts = [chunk["text"] for chunk in chunks]


# Create embeddings
embeddings = model.encode(chunk_texts)


print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)