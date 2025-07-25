import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_and_split_pdfs(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved at faiss_index/")


if __name__ == "__main__":
    # Temporary workaround for macOS OpenMP crash
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    chunks = load_and_split_pdfs("manuals")  # Place your PDFs in a folder named 'manuals'
    build_faiss_index(chunks)
