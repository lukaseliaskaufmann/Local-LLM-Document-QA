import os
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

# --- Strict Prompt ---
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

# --- Regex to detect model names ---
def infer_model_name_from_text(text: str) -> str:
    t = text.upper()
    # Improved regex: catches REYQ plus any underscores, hyphens, alphanumerics
    m = re.search(r"\bREYQ[_A-Z0-9-]*\b", t)
    if m:
        return m.group(0)
    # Fallback to common Daikin codes (FTX, RX, FVX, RXS, FTXS)
    m = re.search(r"\b(FTX|RX|FVX|RXS|FTXS)[\w-]*\b", t)
    return m.group(0) if m else "UNKNOWN"

# --- Streamlit Config ---
st.set_page_config(page_title="Daikin Service Manual Assistant", page_icon="📘")
st.title("📘 Daikin Service Manual Assistant")
st.caption("Ask anything from your service manuals. Everything runs **offline**.")
st.caption("💡 Answers are based only on the actual content in the PDF manuals.")

# --- Initialize chat history in session ---
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

# --- Load vectorstore & debug ---
vs = load_vectorstore()
st.write(
    "Index models:",
    sorted({doc.metadata.get("model_name") for doc in vs.docstore._dict.values()})
)

# --- Sidebar: Model selection ---
st.sidebar.header("🛠️ Model Selection")
mode = st.sidebar.radio("Mode", ["Manual", "Auto-detect"])

selected_model = None
query = None
if mode == "Manual":
    options = sorted({doc.metadata.get("model_name") for doc in vs.docstore._dict.values() if doc.metadata.get("model_name")})
    selected_model = st.sidebar.selectbox("Select Model", options)
    query = st.text_input("Your question")
else:
    query = st.text_input("Your question")
    if query:
        detected = infer_model_name_from_text(query)
        if detected == "UNKNOWN":
            st.sidebar.warning("Could not detect model; please choose manually.")
            options = sorted({doc.metadata.get("model_name") for doc in vs.docstore._dict.values() if doc.metadata.get("model_name")})
            selected_model = st.sidebar.selectbox("Select Model", options)
        else:
            selected_model = detected
            st.sidebar.markdown(f"Detected: `{selected_model}`")

if not selected_model:
    st.warning("Select or detect a Daikin model to continue.")
    st.stop()

# --- Build conversational QA chain ---
def build_chain(model_name: str):
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={
            "filter": {"model_name": model_name},
            "k": 7,
            "fetch_k": 50,
        }
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

chain = build_chain(selected_model)
if not chain:
    st.warning("QA chain init failed. Is Ollama up?")
    st.stop()

# --- Ask & process ---
if query:
    with st.spinner("Thinking..."):
        result = chain({"question": query})
        ans = result.get("answer", "")
        st.success(ans)
        entry = {"question": query, "answer": ans, "sources": []}
        if not ans.lower().startswith("i don't know"):
            docs = result.get("source_documents", [])
            entry["sources"] = docs
            st.markdown("#### Sources:")
            for i, doc in enumerate(docs, 1):
                p = doc.metadata.get("page", "?")
                f = doc.metadata.get("source_file", "?")
                st.info(f"{i}. Page {p} from {f}")
        st.session_state["chat_history"].append(entry)
