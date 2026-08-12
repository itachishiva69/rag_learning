# RAG From Scratch

A simple Retrieval-Augmented Generation (RAG) system built from scratch in Python.

This project is intentionally implemented without LangChain so that each part of the RAG pipeline is clear and understandable.

## What This Project Does

The system:

1. Loads `.txt` documents from the `data/` directory.
2. Splits documents into overlapping chunks.
3. Creates sentence embeddings using `all-MiniLM-L6-v2`.
4. Normalizes the embeddings.
5. Stores them in a FAISS vector index.
6. Embeds a user's query.
7. Retrieves the top-K most similar chunks.
8. Builds context from the retrieved chunks.
9. Sends the context and question to a Groq LLM.
10. Generates an answer grounded in the retrieved context.

## RAG Pipeline

```text
Documents
    ↓
Document Loading
    ↓
Chunking
    ↓
Sentence Embeddings
    ↓
Normalization
    ↓
FAISS Vector Index
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Top-K Retrieved Chunks
    ↓
Context Construction
    ↓
Question + Context
    ↓
Groq LLM
    ↓
Answer
```

## Project Structure

```text
rag_project/
│
├── data/
│   ├── python.txt
│   ├── postgres.txt
│   └── docker.txt
│
├── 01_load.py
├── 02_chunk.py
├── 03_embed.py
├── 04_store.py
├── 05_retrieve.py
├── rag.py
├── main.py
└── README.md
```

### Files

- `01_load.py` - Loads the text documents.
- `02_chunk.py` - Splits documents into overlapping chunks.
- `03_embed.py` - Creates 384-dimensional sentence embeddings.
- `04_store.py` - Creates and populates the FAISS index.
- `05_retrieve.py` - Performs semantic retrieval and builds context.
- `rag.py` - Contains the reusable RAG pipeline functions.
- `main.py` - Runs the complete RAG application.
- `data/` - Contains the source documents.

The numbered files are learning steps. `rag.py` and `main.py` are the refactored version of the complete pipeline.

## Requirements

Recommended Python version:

```text
Python 3.12
```

Main libraries:

```text
sentence-transformers
faiss-cpu
numpy
openai
```

Install them with:

```powershell
pip install sentence-transformers faiss-cpu numpy openai
```

## Groq API

This project uses Groq as the LLM provider through its OpenAI-compatible API.

Set your Groq API key as an environment variable.

### PowerShell

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

Do not hard-code your API key in the Python files or commit it to Git.

The code uses:

```python
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
```

The current example uses:

```text
openai/gpt-oss-20b
```

## Running the Project

From the project directory:

```powershell
python main.py
```

The program will ask:

```text
Ask a question:
```

Example:

```text
What is PostgreSQL used for?
```

The system will retrieve relevant chunks and send them to the LLM.

## Chunking

The current learning implementation uses:

```python
chunk_size = 150
chunk_overlap = 30
```

The next chunk starts at:

```python
start += chunk_size - chunk_overlap
```

For example:

```text
Chunk 1: 0 → 150
Chunk 2: 120 → 270
Chunk 3: 240 → 390
```

The 30-character overlap helps preserve context across chunk boundaries.

This is a simple character-based chunking strategy used for learning. It is not necessarily the best strategy for every real-world document.

## Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

Each chunk is converted into a:

```text
384-dimensional vector
```

If there are 9 chunks:

```text
embeddings.shape
→ (9, 384)
```

The first dimension represents the number of chunks.

The second dimension represents the embedding size.

## FAISS

The project uses:

```python
faiss.IndexFlatIP(384)
```

The embeddings are normalized before being added to the index.

With normalized vectors, inner product can be used to perform cosine-similarity-based retrieval.

During search:

```python
scores, indices = index.search(
    query_embedding,
    k
)
```

`k` specifies how many nearest chunks to retrieve.

The returned indices are then mapped back to the original chunks.

## Metadata

Each chunk currently keeps metadata such as its source file:

```python
{
    "source": "python.txt"
}
```

Conceptually, each stored chunk contains:

```text
Text
+
Embedding
+
Metadata
```

Metadata can later be expanded with information such as:

```text
source
page
document type
date
category
```

This will allow metadata filtering in more advanced versions of the project.

## Current Limitations

This is a learning implementation, not a production RAG system.

Current limitations include:

- Documents are loaded from `.txt` files only.
- Chunking is character-based.
- The FAISS index is rebuilt when the application starts.
- The index is not persisted to disk.
- There is no sophisticated metadata filtering implementation yet.
- There is no reranking.
- There is no query rewriting.
- There is no retrieval evaluation.
- There is no conversational memory.
- The project currently uses a simple prompt for grounding.

These features will be addressed later as part of the advanced RAG learning path.

## Learning Goal

The purpose of this project is to understand RAG from the inside rather than hiding the important components behind a framework.

The core components are:

```text
Document Loading
       ↓
Chunking
       ↓
Embedding
       ↓
Vector Search
       ↓
Retrieval
       ↓
Context Construction
       ↓
LLM Generation
```

After understanding this implementation, the same concepts can be implemented using frameworks such as LangChain.

## Next Step

The next stage of the learning path is **LangChain**.

The goal is not to blindly rewrite this project with LangChain.

Instead, each LangChain abstraction will be mapped to the code already built here:

```text
Our Code                    LangChain
────────────────────────────────────────
load_documents()       →    Document loaders
chunk_documents()      →    Text splitters
create_embeddings()    →    Embeddings
build_index()          →    Vector stores
retrieve()             →    Retrievers
prompt                 →    Prompt templates
generate_answer()      →    Chat models
```

This makes it easier to understand what LangChain is actually doing underneath.
