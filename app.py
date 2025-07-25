import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

st.set_page_config(page_title="Daikin Service Manual Assistant")
st.title("📘 Service Manual Assistant")
st.write("Ask anything from your service manuals (offline, local)")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore(_embeddings):  # underscore avoids unhashable param error
    return FAISS.load_local("faiss_index", _embeddings, allow_dangerous_deserialization=True)


@st.cache_resource
def load_llm():
    try:
        return Ollama(model="mistral")
    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")
        return None

@st.cache_resource
def load_qa_chain():
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    llm = load_llm()
    if llm is None:
        return None
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = load_qa_chain()

query = st.text_input("Your question", placeholder="e.g. My LED is blinking red...")

if query:
    if qa_chain:
        with st.spinner("Thinking..."):
            try:
                answer = qa_chain.run(query)
                st.success(answer)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
    else:
        st.warning("QA chain is not available. Make sure Ollama is running and the model is downloaded.")
