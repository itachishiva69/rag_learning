from src.ingestion import doc_loader, chunk_maker
from src.emmbeddings import create_embedding_model
from src.retrieval import create_vector_store,create_retriever
from src.prompts import create_rag_prompt,format_documents, format_source
from src.llm import create_llm
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

documents = doc_loader('data')

chunks = chunk_maker(documents)

embedding_model = create_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)


retriever = create_retriever(
    vector_store,
    k = 3
    # source = 'data/NIPS-2017-attention-is-all-you-need-Paper.pdf'
)

prompt = create_rag_prompt()

llm = create_llm()

rag_chain = (
    {
        'context' : retriever | RunnableLambda(format_documents),
        'question' : RunnablePassthrough()
    }
    |prompt
    |llm
)
query = input('\n ask a question : ')




response = rag_chain.invoke(query)





# response = llm.invoke(messages)


print('\nAnswer:')
print(response.content)



