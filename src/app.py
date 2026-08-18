from src.ingestion import doc_loader, chunk_maker
from src.emmbeddings import create_embedding_model
from src.retrieval import create_vector_store,create_retriever
from src.prompts import create_rag_prompt,format_documents
from src.llm import create_llm

documents = doc_loader('data')

chunks = chunk_maker(documents)

embedding_model = create_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)


retriever = create_retriever(
    vector_store,
    k = 3,
    source = 'data/NIPS-2017-attention-is-all-you-need-Paper.pdf'
)

query = input('\n ask a question : ')

results = retriever.invoke(query)

context = format_documents(results)

prompt = create_rag_prompt()

messages = prompt.invoke({
    'context' : context,
    'question' : query
})

llm = create_llm()

response = llm.invoke(messages)


print('\nAnswer:')
print(response.content)


