import os
import re
import numpy as np
import faiss

from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ===== macOS OpenMP and threading workarounds =====
os.environ["KMP_DUPLICATE_LIB_OK"]   = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_and_split_pdfs(folder_path: str) -> List:
    """Load each PDF, attach metadata, split into chunks, and return all chunks."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "Error Code", "Electrical Data"],
        chunk_size=500,
        chunk_overlap=50,
    )
    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder_path, file)
        docs = PyMuPDFLoader(path).load()
        match = re.search(r"sm-(reyq[_a-z0-9]+)", file.lower())
        model_name = match.group(1).upper() if match else "UNKNOWN"
        for d in docs:
            d.metadata["source_file"] = file
            d.metadata["model_name"]  = model_name
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    return all_chunks


def dedupe_chunks(chunks: List) -> List:
    """Remove duplicate chunks via a fingerprint of the first 20 words."""
    seen = set()
    deduped = []
    for doc in chunks:
        fingerprint = " ".join(doc.page_content.split()[:20])
        h = hash(fingerprint)
        if h not in seen:
            seen.add(h)
            deduped.append(doc)
    return deduped


def embed_chunks(chunks: List, device: str = "cpu") -> (np.ndarray, HuggingFaceEmbeddings):
    """Embed all chunks at once."""
    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device},
    )
    texts = [doc.page_content for doc in chunks]
    embeds = hf.embed_documents(texts)
    return np.array(embeds), hf


def reduce_dimensionality(vectors: np.ndarray, out_dim: int = 256):
    """Train a FAISS PCA and apply it."""
    d = vectors.shape[1]
    pca = faiss.PCAMatrix(d, out_dim)
    pca.train(vectors)
    reduced = pca.apply_py(vectors)
    return pca, reduced


def build_ivfpq_index(vectors: np.ndarray, nlist: int = 32, m: int = 64, bits: int = 8):
    """Train and build an IVF-PQ index on the given vectors."""
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
    ivfpq.train(vectors)
    ivfpq.add(vectors)
    ivfpq.nprobe = max(1, nlist // 8)
    return ivfpq


if __name__ == "__main__":
    PDF_FOLDER = "manuals"
    INDEX_DIR  = "faiss_index"
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1) Load, split & dedupe
    chunks = load_and_split_pdfs(PDF_FOLDER)
    chunks = dedupe_chunks(chunks)
    print(f"{len(chunks)} chunks after dedupe")

    # 2) Embed
    vectors, hf = embed_chunks(chunks, device="cpu")
    print("Embedded all chunks")

    # 3) PCA reduction (optional)
    pca, reduced = reduce_dimensionality(vectors, out_dim=256)
    faiss.write_VectorTransform(pca, os.path.join(INDEX_DIR, "pca.bin"))
    print("PCA reduction complete")

    # 4) Build IVF-PQ index
    ivfpq = build_ivfpq_index(reduced, nlist=32, m=64, bits=6)
    print("IVF-PQ index built")

    # 5) Wrap PCA + IVF-PQ so that query embeddings are reduced automatically
    index = faiss.IndexPreTransform(pca, ivfpq)

    # 6) Instantiate LangChain FAISS WITHOUT re-embedding
    docstore = {i: chunks[i] for i in range(len(chunks))}
    id_map   = {i: i          for i in range(len(chunks))}

    vs = LCFAISS(
        embedding_function=hf,           # <— use the correct param name here
        index=index,
        docstore=docstore,
        index_to_docstore_id=id_map,
    )
    vs.save_local(INDEX_DIR)

    print(f"Saved vectorstore to {INDEX_DIR} (index.faiss + index.pkl)")
