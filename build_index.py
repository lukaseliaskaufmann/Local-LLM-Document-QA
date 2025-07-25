import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDFs and keep track of page numbers and filenames
def load_and_split_pdfs(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = file
            all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Build FAISS vectorstore from chunks
def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved at faiss_index/")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Temporary fix if needed
    chunks = load_and_split_pdfs("manuals")  # Folder containing your PDF manuals
    build_faiss_index(chunks)
python build_index.py