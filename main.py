from sentence_transformers import SentenceTransformer

from rag import (
    load_documents,
    chunk_documents,
    create_embeddings,
    build_index,
    retrieve,
    build_context,
    generate_answer
)



# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Indexing
documents = load_documents()

chunks = chunk_documents(documents)

embeddings = create_embeddings(
    chunks,
    model
)

index = build_index(embeddings)


# Query
query = input("Ask a question: ")

results = retrieve(
    query,
    model,
    index,
    chunks,
    k=3
)


# Context
context = build_context(results)


# Generation
answer = generate_answer(
    query,
    context
)


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(answer)