from pathlib import Path

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


# Chunk settings
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


print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n" + "=" * 50)
    print("CHUNK:", i)
    print("SOURCE:", chunk["metadata"]["source"])
    print(chunk["text"])