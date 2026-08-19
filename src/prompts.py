from langchain_core.prompts import ChatPromptTemplate

def create_rag_prompt():
    return ChatPromptTemplate.from_template(
        '''
You are a helpful assistant.

Answer the question using only the provided context.

If the answer is not present in the context, say:
'I don't know based on the provided documents.'

Context :
{context}

Question :
{question}
'''
    )


def format_documents(documents):
    return '\n\n'.join(
        document.page_content for document in documents
    )


def format_source(documents):

    sources = []

    for document in documents:
        source = document.metadata.get('source')
        page = document.metadata.get('page')

        sources.append(
            f'{source},page{page+1}'
        )
    return sources