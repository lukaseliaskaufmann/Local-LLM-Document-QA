import os
import streamlit as st
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

# --- Strict Prompt (with history) ---
strict_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are a helpful assistant that answers questions using the provided context from Daikin service manuals.

Use the conversation history below to maintain context.
If the answer is not in the context, say "I don't know."

Conversation history:
{chat_history}

Context:
{context}

Question: {question}
Answer:"""
)

# --- Simple Prompt (no history) ---
simple_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions using the provided context from Daikin service manuals.

Only use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
)

# --- Streamlit App Config ---
st.set_page_config(page_title="Daikin Service Manual Assistant", page_icon="📘")
st.title("📘 Daikin Service Manual Assistant")
st.caption("Ask anything from your service manuals. Everything runs **offline**.")
st.caption("💡 Answers are based only on the actual content in the PDF manuals.")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- Cached Resources ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")}  # or "mps"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    try:
        return Ollama(model="mistral")
    except Exception as e:
        st.error(f"❌ Could not connect to Ollama: {e}")
        return None

# --- Load Vectorstore & Model List ---
vs = load_vectorstore()
model_options = sorted({doc.metadata.get("model_name") for doc in vs.docstore._dict.values() if doc.metadata.get("model_name")})
# Debug print
st.write("Index models:", model_options)

# --- Sidebar: Mode Selection ---
st.sidebar.header("🛠️ Mode Selection")
mode = st.sidebar.radio("Mode", ["Manual", "Ask across all models"])

# --- Mode setup ---
if mode == "Manual":  # Single-model mode
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    st.sidebar.write(f"🔍 Manual mode: searching only {selected_model}")
elif mode == "Ask across all models":  # Ask across all-models mode
    selected_model = None
    st.sidebar.write("🔍 Ask across all models mode: searching all manuals")

# --- Query Input ---
query = st.text_input("Your question")
if not query:
    st.stop()

# --- Build Retrieval QA Chain with Memory ---
def build_chain(model_name: str):
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"filter": {"model_name": model_name}, "k": 7, "fetch_k": 50}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    llm = load_llm()
    if not llm:
        return None
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": strict_prompt},
        return_source_documents=True
    )

# --- Build Simple Retrieval QA Chain (no memory) ---
def build_simple_chain(model_name: str):
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"filter": {"model_name": model_name}, "k": 7, "fetch_k": 50}
    )
    llm = load_llm()
    if not llm:
        return None
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": simple_prompt},
        return_source_documents=True
    )

# --- Ask & Process ---
with st.spinner("🤖 Thinking..."):
    if mode == "Manual":
        chain = build_chain(selected_model)
        if not chain:
            st.warning("QA chain init failed. Is Ollama running?")
        else:
            result = chain({"question": query})
            ans = result.get("answer", "")
            st.subheader(f"Model: {selected_model}")
            if ans.lower().startswith("i don't know"):
                st.info("No information found for this model.")
            else:
                st.success(ans)
                st.markdown("#### Sources:")
                for i, doc in enumerate(result.get("source_documents", []), 1):
                    p = doc.metadata.get("page", "?")
                    f = doc.metadata.get("source_file", "?")
                    st.info(f"{i}. Page {p} from {f}")
        # Save to history
        st.session_state["chat_history"].append({"question": query, "answer": ans})
        
    else:  # Ask across all models
        st.header("🔍 Ask across all models: results for all manuals")
        # display results side by side
        cols = st.columns(len(model_options))
        for col, model in zip(cols, model_options):
            with col:
                st.subheader(f"Model: {model}")
                simple_chain = build_simple_chain(model)
                if not simple_chain:
                    st.warning(f"Chain init failed for {model}.")
                    continue
                result = simple_chain({"query": query})
                ans = result.get("result", "")
                if ans.lower().startswith("i don't know"):
                    st.info("No information found for this model.")
                else:
                    st.success(ans)
                    st.markdown("#### Sources:")
                    for i, doc in enumerate(result.get("source_documents", []), 1):
                        p = doc.metadata.get("page", "?")
                        f = doc.metadata.get("source_file", "?")
                        st.info(f"{i}. Page {p} from {f}")
