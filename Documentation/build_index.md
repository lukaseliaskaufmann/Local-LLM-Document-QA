# 🧠 PDF Manual Vector Indexing with FAISS + LangChain

This project provides a complete pipeline to **process PDF manuals**, **split them into chunks**, **embed them into vectors**, and **index them using FAISS IVF-PQ** for fast, semantic search using [LangChain](https://python.langchain.com).

It is designed for use cases like:

- Embedding **technical documentation** or **equipment manuals**
- Creating a **retrieval-augmented generation (RAG)** system
- Rapid, low-memory **semantic search** over many PDFs

---

## 📌 Overview

The pipeline performs the following steps:

1. **PDF Parsing**: Load documents with metadata.
2. **Chunking**: Smart text splitting with domain-specific separators.
3. **Deduplication**: Remove near-duplicate chunks using lightweight hashing.
4. **Embedding**: Embed all text chunks using `sentence-transformers/all-mpnet-base-v2`.
5. **Indexing**: Build a FAISS IVF-PQ index for compressed ANN search.
6. **Persistence**: Save the index and metadata in LangChain-compatible format.

---

## 🛠️ Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

### Required Python Packages

- `faiss-cpu` – for fast similarity search
- `PyMuPDF` – PDF parsing
- `langchain`, `langchain-community` – document processing and vector store
- `langchain-huggingface` – HuggingFace model wrappers
- `sentence-transformers` – transformer-based embeddings
- `numpy` – numerical operations

---

## 🗂 Folder Structure

```
.
├── manuals/             # Input folder containing PDF files
├── faiss_index/         # Output folder for FAISS + LangChain index
├── main.py              # Main script
├── requirements.txt     # Python dependencies
└── README.md            # You're here
```

---

## ⚙️ Usage

Place your PDF manuals in the `manuals/` folder. Then run:

```bash
python main.py
```

### Output

After running, you will find:

```
faiss_index/
├── index.faiss   # FAISS binary index file
└── index.pkl     # LangChain metadata for document retrieval
```

These files can be loaded later into a LangChain retriever.

---

## 🔍 Step-by-Step Logic Breakdown

### 1. Load & Chunk PDFs

```python
load_and_split_pdfs(folder_path: str) -> List
```

- Uses `PyMuPDFLoader` to parse PDFs.
- Applies custom separators: `"\n\n", "\n", "Error Code", "Electrical Data"` for intelligent chunking.
- Extracts a model identifier (e.g. `SM-REYQ123`) from filenames.
- Attaches metadata like `source_file` and `model_name`.

### 2. Deduplicate Chunks

```python
dedupe_chunks(chunks: List) -> List
```

- Removes similar chunks based on the first 20-word hash fingerprint.
- Prevents redundancy in vector search and reduces index size.

### 3. Embed Text Chunks

```python
embed_chunks(chunks: List, device="cpu") -> (np.ndarray, HuggingFaceEmbeddings)
```

- Uses HuggingFace embedding model: `all-mpnet-base-v2`
- Converts all chunk text into dense vector embeddings.

### 4. Build IVF-PQ FAISS Index

```python
build_ivfpq_index(vectors: np.ndarray, nlist=32, m=64, bits=6) -> faiss.IndexIVFPQ
```

- Trains a compressed FAISS IVF-PQ index:
  - `nlist`: number of Voronoi cells (inverted lists)
  - `m`: number of subquantizers (codebooks)
  - `bits`: bits per subvector
- Enables fast approximate nearest neighbor (ANN) search.

### 5. Create LangChain Vector Store

```python
LCFAISS(
    embedding_function=hf,
    index=index,
    docstore={i: chunks[i] for i in range(len(chunks))},
    index_to_docstore_id={i: i for i in range(len(chunks))}
)
```

- Wraps the FAISS index in a LangChain-compatible interface.
- Stores the text chunks and metadata alongside the vector index.

---

## 📥 Example PDF File Naming

Filenames should follow a recognizable pattern, e.g.:

```
SM-REYQ123_Manual.pdf
SM-REYQ456_Manual_v2.pdf
```

The script extracts `REYQ123` and `REYQ456` as `model_name` metadata.
