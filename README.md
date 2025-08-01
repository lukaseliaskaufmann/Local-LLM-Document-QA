# 📘 Daikin Service Manual Assistant

This is an **offline AI-powered assistant** for querying Daikin service manuals using local PDFs. The assistant uses **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Ollama (Mistral)** to answer technical questions based on indexed content from your PDF manuals.

> 🚀 Ask detailed questions, get grounded answers—100% offline.

---


## 🔧 Features

- **Offline LLM access** via [Ollama](https://ollama.com/) and the `mistral` model
- **PDF ingestion**, chunking, and metadata tagging
- **Vector embedding** using `sentence-transformers/all-mpnet-base-v2`
- **FAISS IVFPQ indexing** for fast retrieval
- **Two QA modes**:
  - Manual (search one model only)
  - Global (search across all manuals)
- **Context-aware conversations** using memory
- Fully **runs offline** — local models, local documents

---
