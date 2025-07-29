import os
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def infer_model_name_from_filename(filename):
    # Example: sm-reyq_a-(sius372201e).pdf -> REYQ_A
    match = re.search(r"sm-(reyq[_a-z0-9]+)", filename.lower())
    return match.group(1).upper() if match else "UNKNOWN"


def load_and_split_pdfs(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            model_name = infer_model_name_from_filename(file)

            for doc in docs:
                doc.metadata["source_file"] = file
                doc.metadata["model_name"] = model_name

            all_docs.extend(docs)

    print(f"✅ Loaded {len(all_docs)} total pages from {folder_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    folder_path = "manuals"  # Your folder with PDFs
    chunks = load_and_split_pdfs(folder_path)
    build_faiss_index(chunks)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}  # or "mps"
)

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
docs = vectorstore.similarity_search("test", k=10)

models_found = set(doc.metadata.get("model_name", "none") for doc in docs)
print("Models in FAISS metadata:", models_found)