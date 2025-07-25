# app.py
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Streamlit UI configuration
st.set_page_config(page_title="Service Manual Assistant", page_icon="📘")
st.title("📘 Service Manual Assistant")
st.write("Ask anything from your indexed service manuals. Responses will stream as they arrive.")

# Sidebar for retrieval settings
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of docs to retrieve (k)", 1, 20, 10)

# Cache embeddings
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
load_embeddings = st.cache_resource(load_embeddings)

# Cache FAISS vector store
def load_vectorstore(_embeddings):
    index_dir = "faiss_index"
    if not os.path.isdir(index_dir):
        st.error("FAISS index not found. Please run `index.py` first to generate the index.")
        return None
    return FAISS.load_local(
        index_dir,
        _embeddings,
        allow_dangerous_deserialization=True
    )
load_vectorstore = st.cache_resource(load_vectorstore)

# Cache Ollama LLM
def load_llm():
    try:
        return Ollama(model="mistral")
    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}. Ensure Ollama is running locally.")
        return None
load_llm = st.cache_resource(load_llm)

# Initialize components
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm()
qa_chain = None
if vectorstore and llm:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# User input
query = st.text_input("Your question:", key="manual_query")
if query:
    if not qa_chain:
        st.error("QA chain not available. Please check your index and Ollama setup.")
    else:
        st.chat_message("user").write(query)
        container = st.chat_message("assistant")
        stream_handler = StreamlitCallbackHandler(parent_container=container)

        # Debug: show retrieved snippets
        docs = retriever.get_relevant_documents(query)
        st.write(f"🔍 Retrieved {len(docs)} documents for query: '{query}'")
        for i, doc in enumerate(docs[:3]):
            snippet = doc.page_content.replace("\n", " ")[:200]
            st.write(f"Doc {i+1} snippet: {snippet}...")

        # Stream the answer
        try:
            result = qa_chain({"query": query}, callbacks=[stream_handler])
            if result and "source_documents" in result:
                st.markdown("#### Sources:")
                for i, src in enumerate(result["source_documents"]):
                    file = src.metadata.get("source_file", "unknown file")
                    page = src.metadata.get("page", "unknown page")
                    st.info(f"Source {i+1}: `{file}` (page {page})")
        except Exception as e:
            st.error(f"Failed to get an answer: {e}")

