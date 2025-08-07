# 📘 Daikin Service Manual Assistant (Offline Semantic QA with Streamlit + LangChain)

A local-first, offline-capable Streamlit application that allows you to **ask technical questions about Daikin service manuals** using semantic search over PDFs. This app combines **FAISS**, **LangChain**, **HuggingFace embeddings**, and **Ollama-powered LLMs** to create an intelligent, privacy-respecting assistant that runs entirely on your machine.

---

## ⚡ What This Does

✅ Loads and parses PDF manuals
✅ Splits content into text chunks with smart separators
✅ Embeds those chunks using Sentence Transformers
✅ Stores and filters vectors using FAISS with metadata
✅ Loads the vector index into LangChain
✅ Runs a semantic search and feeds the context to a local LLM
✅ Supports chat history in Manual mode (multi-turn)
✅ Fully offline: no OpenAI, no external calls

---

## 📁 Project Structure

```
.
├── main.py               # Streamlit app (this README is for this)
├── build_index.py      # Script to create embeddings and FAISS index
├── faiss_index/          # Saved FAISS vector store (index.faiss + index.pkl)
├── manuals/              # Folder of input PDF manuals
├── requirements.txt
└── README.md
```

---

## 🔍 How the App Works (In-Depth)

### 📥 1. Vectorstore Loading

```python
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
```

- Loads the saved FAISS index (`index.faiss`) and LangChain docstore (`index.pkl`).
- The vectorstore includes:
  - The FAISS index itself (approximate nearest neighbor engine)
  - A **docstore**: a Python dict mapping vector IDs → `Document` objects (with `.page_content` and `.metadata`)
  - A mapping from FAISS internal IDs to document keys

> **Security Note**: `allow_dangerous_deserialization=True` allows loading Python-pickled files — only use this with **trusted, local** files.

---

---

### 📄 2. Metadata in Use

When embedding documents, the script extracts and attaches the following metadata to each chunk:

| Metadata Key    | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `model_name`  | Parsed from the filename (`SM-REYQ123_Manual.pdf` → `REYQ123`) |
| `source_file` | Original filename of the PDF                                        |
| `page`        | Page number in the source PDF                                       |

This metadata is attached to each `Document` object and enables precise filtering during vector search. For example, to restrict results to a specific model:

```python
search_kwargs = {
    "filter": {"model_name": selected_model},  # only search chunks from a specific model
    "k": 7,            # return the top 7 most relevant results
    "fetch_k": 50      # initially retrieve 50 approximate matches before filtering/scoring
}
```

📌 **Filtering is applied before scoring**, which means FAISS retrieves vectors that match the `model_name`, then narrows down to the top `k` based on similarity scores.

---

### 🧠 Why `k=7` and `fetch_k=50`?

- `k` (`top_k`): Number of final context documents passed to the LLM. In this app, we use **7** chunks to give the LLM enough information without overwhelming context limits.
- `fetch_k`: Number of initial vectors retrieved from FAISS before filtering by metadata. A **higher value improves recall**, especially when filtering is active (e.g., per model). Without enough `fetch_k`, you may miss relevant chunks due to early truncation by FAISS.

> Think of `fetch_k` as a broader net, and `k` as the focused shortlist that’s actually used to answer your question.

💡 You can tune these values:

- Increase `fetch_k` if you have large PDFs or observe missing context.
- Reduce `k` if you hit token length issues with the LLM (especially on smaller models).

### 🧠 3. Embeddings

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}  # or "cuda", "mps"
)
```

- Embeds all text chunks into 768-dim vectors
- Embedding model is cached for performance via `@st.cache_resource`

The `embed_manuals.py` script generates and saves the vectorstore to disk.

---

### 🔁 4. Vectorstore Wrapping

LangChain wraps the FAISS index and the docstore to provide a `Retriever` interface.

If the docstore is just a dict, it’s wrapped in a `search()`-compatible interface:

```python
if isinstance(vs.docstore, dict):
    class _DocstoreWrapper(dict):
        def search(self, key):
            return super().__getitem__(key)
    vs.docstore = _DocstoreWrapper(vs.docstore)
```

This allows LangChain retrievers to function normally even if the docstore was deserialized into a plain dict.

---

### 📚 5. LLM Setup

The app uses `Ollama` to run a local model (e.g., `mistral`):

```python
from langchain_community.llms import Ollama

def load_llm():
    return Ollama(model="mistral")
```

This lets you use a fully local language model for inference without an internet connection.

---

## 🧠 Two Modes of Question Answering

### 🧾 1. Manual Mode (Per Model)

```python
# Example: filter by model_name = "REYQ123"
retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"filter": {"model_name": "REYQ123"}, "k": 7}
)
```

- Uses **ConversationalRetrievalChain**
- Maintains memory of chat history (`ConversationBufferMemory`)
- Ideal for step-by-step technical troubleshooting

```python
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=ConversationBufferMemory(...),
    combine_docs_chain_kwargs={"prompt": strict_prompt}
)
```

### 🌐 2. Ask Across All Models

- No filtering — each model gets a fresh answer
- Stateless (no memory)
- Uses `RetrievalQA` chain with `simple_prompt`

```python
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": simple_prompt}
)
```

---

## 🧠 Prompt Templates

### Strict Prompt (Manual mode)

```text
You are a helpful assistant that answers questions using the provided context from Daikin service manuals.
Give long detailed answers.

Use the conversation history below to maintain context.
If the answer is not in the context, say "I don't know."
```

### Simple Prompt (All Models mode)

```text
Only use the context below to answer the question.
If the answer is not in the context, say "I don't know."
```

Both prompts ensure grounded answers based on retrieved text chunks.

---

## 🖥 How to Run

1. Make sure your FAISS index is built and saved to `faiss_index/`
2. Start Ollama:

```bash
ollama run mistral
```

3. Launch the Streamlit app:

```bash
streamlit run main.py
```

---

## 📌 Example File Name Conventions

For best results, follow this pattern when naming PDFs:

```
SM-REYQ456_Manual.pdf
```

This allows the app to extract `REYQ456` as the `model_name` metadata for later filtering.

---

## 📦 Example Output

### Input:

```text
What does error code U4 mean on model REYQ123?
```

### Output:

```
U4 indicates a communication error between the outdoor and indoor units...
```

> Source: Page 14 of SM-REYQ123_Manual.pdf

---

## 🔐 Security Warning

This app uses:

```python
allow_dangerous_deserialization=True
```

This allows loading Python pickles for the FAISS index metadata. Only use this with files you trust and never over the web.

---

## 🧾 requirements.txt (example)

```txt
streamlit
langchain
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
PyMuPDF
```

---

## 📚 Useful Extensions

- 🔄 Add document upload interface to embed new manuals on the fly
- 🔎 Add keyword search fallback if semantic fails
- 🔧 Add PDF preview of source documents
- 🤝 Integrate with RAG pipelines or custom agents

---

## 👨‍💻 Author

Built by [Your Name or Org]
License: MIT — free to use, modify, and distribute

---
