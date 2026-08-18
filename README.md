# RAG Learning Project

A practical Retrieval-Augmented Generation (RAG) project built with **Python, LangChain, Hugging Face embeddings, FAISS, and Groq**.

This project is being developed step by step to understand how a production-style RAG system works, starting from document ingestion and ending with an LLM-generated answer based on retrieved document context.

## Architecture

```text
PDF Documents
     ↓
PDF Loader
     ↓
Document Objects
     ↓
Text Chunking
     ↓
Embedding Model
     ↓
FAISS Vector Store
     ↓
Retriever
     ↓
Relevant Documents
     ↓
Context Construction
     ↓
Prompt Template
     ↓
Groq LLM
     ↓
Generated Answer
```

## Project Structure

```text
rag_learning/
│
├── data/
│   ├── NIPS-2017-attention-is-all-you-need-Paper.pdf
│   ├── UnderstandingDeepLearning_02_09_26_C.pdf
│   └── LLM.pdf
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── prompts.py
│   └── llm.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Current Dataset

| Document | Pages | Chunks |
|---|---:|---:|
| Attention Is All You Need | 11 | 76 |
| Understanding Deep Learning | 541 | 3,131 |
| LLM | 47 | 617 |
| **Total** | **599** | **3,824** |

The current chunking configuration is:

```python
chunk_size=500
chunk_overlap=50
```

## 1. PDF Ingestion

PDFs are loaded using LangChain's `PyPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/document.pdf")
documents = loader.load()
```

Each page becomes a LangChain `Document`.

A document contains:

```python
Document(
    page_content="...",
    metadata={
        "source": "...",
        "page": 0,
        ...
    }
)
```

The metadata is preserved when documents are split into chunks.

### PDF limitation

Standard PDF text extraction primarily extracts the document's text layer. Images, diagrams, and scanned content may not be represented in `page_content`.

One of the PDFs in this project produced the following extraction warning:

```text
Exceeded 5000 form XObject invocations while extracting text;
further form content is skipped.
```

Handling complex PDFs, OCR, tables, images, and multimodal documents is planned as a later improvement.

## 2. Chunking

Documents are split using:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
```

Chunking is necessary because embedding an entire large document into one vector can mix too much information.

Very small chunks can lose context, while very large chunks can contain excessive irrelevant information.

The goal is to create chunks that contain enough coherent information for useful retrieval.

## 3. Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through LangChain's Hugging Face integration.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The current dataset produces:

```text
Number of chunks:       3824
Embedding dimension:    384
```

Therefore the conceptual embedding matrix is:

```text
(3824, 384)
```

Each chunk is represented by a 384-dimensional vector.

## 4. FAISS Vector Store

FAISS is used as the vector store.

```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    chunks,
    embedding_model
)
```

FAISS allows the application to perform similarity searches against the embedded chunks.

The project previously implemented raw FAISS manually. The LangChain integration now handles the mapping between:

```text
Document ↔ Embedding ↔ FAISS index
```

## 5. Similarity Search

A query is embedded using the same embedding model and compared against the stored vectors.

Example:

```text
"What is the Transformer architecture?"
```

The system retrieves semantically related chunks.

The test retrieval successfully returned relevant content from:

```text
Understanding Deep Learning
Attention Is All You Need
LLM
```

## 6. Retriever

The FAISS vector store can be converted into a LangChain retriever:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

The query can then be executed with:

```python
documents = retriever.invoke(query)
```

The retriever returns LangChain `Document` objects.

```python
document.page_content
document.metadata
```

The retriever provides an abstraction over the underlying vector store.

## 7. Metadata Filtering

Chunks contain metadata such as:

```python
{
    "source": "data/LLM.pdf",
    "page": 20
}
```

Metadata filtering can restrict which documents participate in the search.

For example:

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {
            "source": "data/LLM.pdf"
        }
    }
)
```

The conceptual process becomes:

```text
All chunks
    ↓
Metadata filter
    ↓
Eligible chunks
    ↓
Semantic search
    ↓
Top-k chunks
```

Metadata filtering does not replace semantic search.

## 8. Context Construction

The retriever returns `Document` objects, but the LLM needs the actual text.

The retrieved `page_content` values are combined into a context:

```python
context = "\n\n".join(
    document.page_content
    for document in documents
)
```

Conceptually:

```text
Document 1
    ↓
page_content

Document 2
    ↓
page_content

Document 3
    ↓
page_content

       ↓

Combined Context
```

## 9. Prompt Template

A reusable LangChain prompt is created using `ChatPromptTemplate`.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.

    Answer the question using only the provided context.

    If the answer is not present in the context, say:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}
    """
)
```

The placeholders:

```text
{context}
{question}
```

are filled when the prompt is invoked.

## 10. Groq LLM

The project uses Groq for answer generation.

The API key is stored in `.env`:

```text
GROQ_API_KEY=your_api_key
```

The `.env` file should never be committed to Git.

The current accessible model used by the project is:

```text
openai/gpt-oss-20b
```

Example:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
```

## 11. Complete RAG Flow

The current application performs:

```python
query = input("Ask a question: ")

documents = retriever.invoke(query)

context = "\n\n".join(
    document.page_content
    for document in documents
)

messages = prompt.invoke({
    "context": context,
    "question": query
})

response = llm.invoke(messages)

print(response.content)
```

The complete flow is:

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
page_content
      ↓
Context
      ↓
Prompt Template
      ↓
Groq LLM
      ↓
Answer
```

## Advanced RAG Concepts Covered

The project is also being developed around the following advanced RAG concepts.

### Better Chunking

Creating chunks that preserve meaningful context while reducing irrelevant information.

### Metadata Filtering

Restricting retrieval to documents matching structured metadata.

### Query Rewriting

Rewriting context-dependent questions into standalone retrieval queries.

Example:

```text
"What about authentication?"
            ↓
"What authentication mechanisms does PostgreSQL support?"
```

### Hybrid Search

Combining:

```text
Semantic Search
+
Keyword Search
```

Semantic search is useful for meaning and concepts, while keyword search is useful for exact terms such as error codes, identifiers, and technical names.

### Reranking

The retriever finds candidate chunks, then a reranker evaluates those candidates more carefully and improves their ordering.

```text
Retriever
   ↓
50 candidates
   ↓
Reranker
   ↓
Top relevant chunks
```

### Context Compression

Removing irrelevant portions from retrieved documents before sending the context to the LLM.

```text
Retrieved Documents
        ↓
Context Compression
        ↓
Relevant Information
        ↓
LLM
```

### Conversational RAG

Using conversation history to handle follow-up questions.

```text
Conversation History
        +
Current Question
        ↓
Query Rewriting
        ↓
Retriever
        ↓
Context
        ↓
LLM
```

### Self-Query Retrieval

Allowing a query-understanding step to extract structured metadata filters from natural-language questions.

For example:

```text
"What did the 2017 paper say about attention?"
```

can conceptually become:

```text
Metadata filter:
year = 2017

Semantic query:
"attention"
```

### RAG Evaluation

Evaluating different parts of the RAG system independently:

- Retrieval recall
- Retrieval precision
- Answer quality
- Faithfulness / groundedness

## Learning Progress

```text
RAG Foundations                   ✅
Document Processing               ✅
Embeddings                        ✅
Vector Search                     ✅
Raw FAISS                         ✅
RAG From Scratch                  ✅
LangChain Fundamentals            ✅
LangChain FAISS                   ✅
Retriever                         ✅
Advanced RAG Theory               ✅
PDF-based RAG Implementation      🚧
End-to-End LangChain RAG          ✅
LCEL RAG Chain                    ⬜
Advanced Retrieval Implementation ⬜
Production RAG                    ⬜
Agentic RAG                       ⬜
LangGraph                         ⬜
Production RAG Agent              ⬜
```

## Future Roadmap

```text
Current RAG
    ↓
LCEL RAG Chain
    ↓
Dynamic Metadata Filtering
    ↓
Query Rewriting
    ↓
Hybrid Search
    ↓
Reranking
    ↓
Context Compression
    ↓
Conversational RAG
    ↓
RAG Evaluation
    ↓
Production RAG
    ↓
Tool Calling
    ↓
RAG as a Tool
    ↓
Agents
    ↓
LangGraph
    ↓
Agentic RAG
    ↓
Production RAG Agent
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```text
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
python -m src.app
```

## Security

Never commit API keys.

Add `.env` to `.gitignore`:

```text
.env
```

Check before committing:

```bash
git status
```

## Project Goal

This project is intentionally being built from the fundamentals upward.

The goal is not simply to use LangChain APIs. The goal is to understand what happens underneath them:

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Search
    ↓
Retrieval
    ↓
Context
    ↓
Prompt
    ↓
LLM
```

The project will then progressively turn this basic RAG system into a more capable **production-style RAG and Agentic RAG system**.
