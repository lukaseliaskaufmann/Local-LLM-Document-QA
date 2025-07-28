import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- Streamlit Page Config ---
st.set_page_config(page_title="Daikin Service Manual Assistant", page_icon="📘")
st.title("📘 Daikin Service Manual Assistant")
st.caption("Ask anything from your service manuals. Everything runs **offline**.")

# --- Cached Resources ---

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "mps"}  # or "cpu" if MPS isn't stable
    )

@st.cache_resource
def load_vectorstore(_embeddings):
    return FAISS.load_local("faiss_index", _embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    try:
        return Ollama(model="mistral")  # Change to your preferred local model
    except Exception as e:
        st.error(f"❌ Could not connect to Ollama: {e}")
        return None

@st.cache_resource
def load_qa_chain():
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=13)
    llm = load_llm()
    if not llm:
        return None
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# --- Run QA ---

qa_chain = load_qa_chain()

query = st.text_input("🔍 Your question", placeholder="e.g. My LED is blinking red...")

if query:
    if qa_chain:
        with st.spinner("🤖 Thinking..."):
            try:
                result = qa_chain.invoke({"query": query})
                st.success(result["result"])

                # Show sources
                if result.get("source_documents"):
                    st.markdown("#### 📄 Sources:")
                    for i, doc in enumerate(result["source_documents"]):
                        page = doc.metadata.get("page", "unknown")
                        file = doc.metadata.get("source_file", "unknown file")
                        st.info(f"**Source {i+1}:** Page `{page}` from `{file}`")

            except Exception as e:
                st.error(f"❌ Error during query: {e}")
    else:
        st.warning("⚠️ QA chain is not available. Ensure Ollama is running with the required model.")
