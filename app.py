import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Streamlit UI setup
st.set_page_config(page_title="Daikin Service Manual Assistant")
st.title("📘 Service Manual Assistant")
st.write("Ask anything from your service manuals (offline, local)")

# Cache HuggingFace embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cache FAISS vectorstore
@st.cache_resource
def load_vectorstore(_embeddings):
    return FAISS.load_local("faiss_index", _embeddings, allow_dangerous_deserialization=True)

# Cache Ollama LLM
@st.cache_resource
def load_llm():
    try:
        return Ollama(model="mistral")
    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")
        return None

# Cache QA chain setup
@st.cache_resource
def load_qa_chain():
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    llm = load_llm()
    if llm is None:
        return None
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Load chain
qa_chain = load_qa_chain()

# User query input
query = st.text_input("Your question", placeholder="e.g. My LED is blinking red...")

# Run query and show answer with source metadata
if query:
    if qa_chain:
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": query})
                st.success(result["result"])

                if "source_documents" in result:
                    st.markdown("#### Sources:")
                    for i, doc in enumerate(result["source_documents"]):
                        page = doc.metadata.get("page", "unknown")
                        file = doc.metadata.get("source_file", "unknown file")
                        st.info(f"**Source {i+1}**: Page {page} from `{file}`")

            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
    else:
        st.warning("QA chain is not available. Make sure Ollama is running and the model is downloaded.")
