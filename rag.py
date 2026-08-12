from pathlib import Path
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


load_dotenv()

api_key = os.getenv('groq_api')


MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(data_dir="data"):
    documents = []

    for file_path in Path(data_dir).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "text": text,
            "metadata": {
                "source": file_path.name
            }
        })

    return documents


def chunk_documents(
    documents,
    chunk_size=150,
    chunk_overlap=30
):
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

    return chunks


def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    return embeddings


def build_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def retrieve(
    query,
    model,
    index,
    chunks,
    k=3
):
    query_embedding = model.encode([query])

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):
        results.append({
            "score": float(score),
            "text": chunks[index_number]["text"],
            "metadata": chunks[index_number]["metadata"]
        })

    return results


def build_context(results):
    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['metadata']['source']}\n"
            f"Content: {result['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, context):
    prompt = f"""
You are a helpful assistant.

Answer the question using only the provided context.
If the answer cannot be found in the context,
say that you don't have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    return response.output_text