from langchain_community.vectorstores import FAISS

def create_vector_store(chunks,embedding_model):
    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )
    return vector_store



# def search_documents(vector_store,query,k=3):
#     results = vector_store.similarity_search(
#         query,
#         k = k
#     )
#     return results



def create_retriever(vector_store,k=3,source=None):

    search_kwargs = {
        'k':k
    }
    if source is not None:
        search_kwargs['filter'] = {
            'source' : source
        }

    return vector_store.as_retriever(
        search_kwargs = search_kwargs
    )