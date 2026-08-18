from langchain_huggingface import HuggingFaceEmbeddings


# creating embeddings for chunks

def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    )


