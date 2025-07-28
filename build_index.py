import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load and split PDFs into chunks
def load_and_split_pdfs(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = file
            all_docs.extend(docs)
    print(f"✅ Loaded {len(all_docs)} total pages from {folder_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# Build and save FAISS vectorstore
def build_faiss_index(chunks, device="mps"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device}
        )
        print(f"✅ Loaded embedding model on `{device}`")
    except Exception as e:
        print(f"❌ Failed to load on `{device}`, falling back to CPU. Error: {e}")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index saved to `faiss_index/`")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for macOS libiomp issue
    folder_path = "manuals"  # Your folder with PDFs
    chunks = load_and_split_pdfs(folder_path)
    build_faiss_index(chunks)
