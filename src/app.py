from src.ingestion import doc_loader, chunk_maker
from src.emmbeddings import create_embedding_model
from src.retrieval import create_vector_store, create_retriever
from src.prompts import (
    create_rag_prompt,
    format_documents,
    rewrite_promt
)
from src.llm import create_llm
from langchain_core.runnables import RunnableLambda


# -----------------------------------------
# 1. Load documents
# -----------------------------------------

documents = doc_loader("data")

chunks = chunk_maker(documents)


# -----------------------------------------
# 2. Create embedding model
# -----------------------------------------

embedding_model = create_embedding_model()


# -----------------------------------------
# 3. Create vector store
# -----------------------------------------

vector_store = create_vector_store(
    chunks,
    embedding_model
)


# -----------------------------------------
# 4. Create retriever
# -----------------------------------------

retriever = create_retriever(
    vector_store,
    k=3
)


# -----------------------------------------
# 5. Create prompt and LLM
# -----------------------------------------

prompt = create_rag_prompt()

llm = create_llm()


# -----------------------------------------
# 6. Query rewriting chain
# -----------------------------------------

rewrite_chain = (
    rewrite_promt
    | llm
    | RunnableLambda(lambda response: response.content)
)


# -----------------------------------------
# 7. Retrieval chain
# -----------------------------------------

retrieval_chain = (
    rewrite_chain
    | retriever
)


# -----------------------------------------
# 8. Conversational RAG chain
# -----------------------------------------

conversational_rag_chain = (
    {
        "context": (
            retrieval_chain
            | RunnableLambda(format_documents)
        ),

        "question": RunnableLambda(
            lambda x: x["question"]
        )
    }
    | prompt
    | llm
)


# -----------------------------------------
# 9. Format conversation history
# -----------------------------------------

def format_history(history):

    formatted_history = []

    for message in history:

        role = message["role"]
        content = message["content"]

        formatted_history.append(
            f"{role}: {content}"
        )

    return "\n".join(formatted_history)


# -----------------------------------------
# 10. Store conversation
# -----------------------------------------

history = []


# -----------------------------------------
# 11. Conversation loop
# -----------------------------------------

print("\nRAG chatbot started.")
print("Type 'exit' or 'quit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break


    # Convert previous messages into text
    formatted_history = format_history(history)


    # Invoke conversational RAG
    response = conversational_rag_chain.invoke({
        "history": formatted_history,
        "question": question
    })


    # Print answer
    print("\nAI:")
    print(response.content)
    print()


    # Save current user message
    history.append({
        "role": "user",
        "content": question
    })


    # Save AI response
    history.append({
        "role": "assistant",
        "content": response.content
    })