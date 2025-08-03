# 📘 Daikin Service Manual Assistant

**An offline, LLM-powered semantic search and QA system** for querying Daikin service manuals via local PDF documents. This assistant is architected for high-performance technical document retrieval and reasoning using open-source AI components — fully air-gapped, no cloud dependencies.

---

## 🧠 System Architecture Overview

This tool leverages modern NLP techniques to enable efficient, context-aware interactions with complex technical documentation:

- ✅ **PDF Ingestion & Chunking**  
  Parses Daikin manuals into context-preserving text segments with optional metadata tagging (e.g., section headers, model numbers)

- ✅ **Vector Embedding**  
  Encodes all chunks using `sentence-transformers/all-mpnet-base-v2`, optimized for semantic similarity in dense vector space

- ✅ **FAISS IVFPQ Indexing**  
  Uses **inverted file + product quantization** for fast, memory-efficient approximate nearest-neighbor search on large corpora

- ✅ **Local LLM Inference**  
  Uses [Ollama](https://ollama.com/) with the `mistral` model for on-device natural language responses — no API calls, no telemetry

- ✅ **LangChain Orchestration**  
  Manages the document retrieval, memory, and query-routing pipeline

- ✅ **Dual QA Modes**  
  - 🔍 **Manual-Specific QA** — restricts vector search to a selected document  
  - 🌐 **Global QA** — searches across all indexed manuals simultaneously

- ✅ **Contextual Memory**  
  Maintains conversational memory for multi-turn reasoning and follow-up questions

---

## 🔐 100% Offline & Private

All components — including embedding, indexing, and LLM inference — run entirely **on your local machine**.  
There is **no internet connection**, **no cloud API calls**, and **no third-party data sharing** at any point in the workflow.

This ensures:

- 🔒 **Maximum data protection** — no customer, service, or technical data ever leaves your environment  
- 🧭 **Full control over documents, queries, and model behavior**  
- 🛠️ Ideal for sensitive use cases in **field service, internal support, or compliance-restricted environments**

---

## 🚀 Example Use Cases

- 🔧 Diagnose fault codes by model and region  
- 🔍 Retrieve parameter adjustment procedures  
- 📋 Lookup wiring diagrams or component specs  
- 🧩 Ask complex, multi-step service queries across manuals

---

## 🧱 Stack Summary

| Component         | Description                                  |
|------------------|----------------------------------------------|
| **LLM**          | `mistral` via [Ollama](https://ollama.com/)  |
| **Embeddings**   | `all-mpnet-base-v2` (Hugging Face)           |
| **Vector DB**    | FAISS with IVFPQ indexing                    |
| **Pipeline**     | LangChain (local mode)                       |
| **Input Format** | PDF service manuals                          |
| **Output**       | Textual QA with source-grounded responses    |

