from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# loading documents from file destination


from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def doc_loader(data_path="data"):
    documents = []

    pdf_files = list(Path(data_path).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files:")

    for file_path in pdf_files:
        print(f"  {file_path}")

        loader = PyPDFLoader(str(file_path))
        loaded_docs = loader.load()

        print(f"  Pages loaded: {len(loaded_docs)}")

        documents.extend(loaded_docs)

    return documents
# def doc_loader(doc_location):
#     documents = []
#     for file_path in Path(doc_location).glob('*.pdf'):
#         loader = PyPDFLoader(
#             str(file_path)
#         )
#         print(f'{file_path} is loaded')
#         documents.extend(loader.load())
#         return documents

# creating chunks for the loaded documents

def chunk_maker(documents:list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(documents)
    return chunks




 

if __name__ == "__main__":
    documents = doc_loader('../data')
    chunks = chunk_maker(documents)

    for i, chunk in enumerate(chunks,start = 1):
        print(f'chunk {i}')
        print(chunk.metadata)
        print(chunk.page_content[:50])

