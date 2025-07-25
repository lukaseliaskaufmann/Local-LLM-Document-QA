# index.py
import os
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_and_split_pdfs(folder_path: str):
    all_docs = []
    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder_path, file)
        try:
            loader = PyMuPDFLoader(path)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {file}")
            for idx, doc in enumerate(docs):
                doc.metadata["source_file"] = file
                doc.metadata["page"] = idx + 1
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    print(f"Total pages loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_faiss_index(chunks, index_path: str = "faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved at {index_path}/")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser(
        description="Build FAISS index for service manuals"
    )
    parser.add_argument(
        "--folder", type=str, default="manuals",
        help="Path to folder containing PDF manuals"
    )
    parser.add_argument(
        "--index-path", type=str, default="faiss_index",
        help="Directory to save the FAISS index"
    )
    args = parser.parse_args()

    chunks = load_and_split_pdfs(args.folder)
    build_faiss_index(chunks, args.index_path)
