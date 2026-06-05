import streamlit as st
import requests
import json
import os
import traceback
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sia - SSSIHL Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)

if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-box'>
    <h1>🎓 Sia — SSSIHL Knowledge Assistant (Visitor)</h1>
    <p>Ask questions about admissions, campus, and programs. Powered by OpenRouter free gateway and Pinecone RAG.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tokens" not in st.session_state:
    st.session_state.tokens = 0
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

# ── Initialize embeddings, Pinecone index and OpenRouter key ────────────────────
@st.cache_resource
def init_rag():
    try:
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        index_name = st.secrets.get("PINECONE_INDEX", "saiinst")

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)

        return embeddings, index, openrouter_key
    except Exception as e:
        st.error("⚠️ Error loading API keys. Set `OPENROUTER_API_KEY` and `PINECONE_API_KEY` in Streamlit secrets.")
        st.stop()

with st.spinner("🔄 Initializing system and connecting to Pinecone..."):
    embeddings, index, openrouter_key = init_rag()

# Quick OpenRouter connectivity check
def check_openrouter_key(key):
    try:
        headers = {"Authorization": f"Bearer {key}"}
        r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        return r.status_code, (r.text[:1000] if r.text else "")
    except Exception as e:
        return None, str(e)

status, body = check_openrouter_key(openrouter_key)
if status is None:
    st.warning(f"OpenRouter diagnostics failed: {body}")
elif status != 200:
    st.error(f"OpenRouter returned HTTP {status}. Response: {body}")
else:
    st.info("OpenRouter models endpoint reachable.")

# ── Settings ──────────────────────────────────────────────────────────────────
selected_model = "openrouter/free"
top_k = 5
min_score = 0.45

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=selected_model,
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    max_tokens=1024,
    default_headers={
        "HTTP-Referer": "https://sssihl.edu.in",
        "X-Title": "SSSIHL Knowledge Assistant"
    }
)

# ── Prompt ───────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are Sia, a helpful AI assistant for Sri Sathya Sai Institute of Higher Learning (SSSIHL).
You are interacting with a visitor.

Here is some retrieved information from the institute's database:
---
{context}
---

Question: {question}
Answer:
""")

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str):
    try:
        vec = embeddings.embed_query(query)
        results = index.query(vector=vec, top_k=top_k, include_metadata=True)
        parts, sources = [], []
        for m in results.get("matches", []):
            if m.get("score", 0) < min_score:
                continue
            text = m.get("metadata", {}).get("text", "")[:2000]
            src = m.get("metadata", {}).get("source_file", "doc")
            pg = m.get("metadata", {}).get("page", "?")
            parts.append(text)
            sources.append(f"{src} p.{pg}")
        return "\n\n---\n\n".join(parts), list(dict.fromkeys(sources))
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        print(traceback.format_exc())
        return "", []

def ask(question: str):
    history = "\n".join([f"User: {m['content']}" if m['role']=='user' else f"Bot: {m['content']}" for m in st.session_state.messages[-6:]])
    context, sources = retrieve(f"{history}\n{question}")
    if not context:
        context = "No relevant documents found."

    try:
        response = llm.invoke(
            PROMPT.format_messages(
                context=context,
                question=question
            )
        )
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            st.session_state.tokens += response.usage_metadata.get("total_tokens", 0)
        return response.content, sources
    except Exception as e:
        st.error(f"Model error: {e}")
        print(traceback.format_exc())
        return f"⚠️ Error from model: {e}", []

# ── Sidebar (visitor only) ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Sia @ SSSIHL")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.msg_count = 0
        st.session_state.tokens = 0
        st.experimental_rerun()
    st.markdown("---")
    st.markdown("This app uses the OpenRouter free gateway and Pinecone for retrieval.")

# ── Suggestion chips ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    c1, c2, c3, c4 = st.columns(4)
    chips = {
        c1: "What programs does SSSIHL offer?",
        c2: "Tell me about the admission process",
        c3: "What is integral education at SSSIHL?",
        c4: "Describe the campus facilities"
    }
    for col, text in chips.items():
        if col.button(text, use_container_width=True):
            st.session_state["pending"] = text
            st.experimental_rerun()

# ── Chat display ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👤 &nbsp;{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'><b>Sia</b> &nbsp;{msg['content']}</div>", unsafe_allow_html=True)

# ── Chat input ──────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about SSSIHL...")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.msg_count += 1
    with st.spinner("🧠 Searching documents and generating response..."):
        answer, sources = ask(question)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "model_used": selected_model})
    st.experimental_rerun()
