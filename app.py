import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI                     # ✅ changed from langchain_groq
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SSSIHL Knowledge Assistant",
    page_icon="🕉️",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f0f4f9; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #004c97 0%, #002d62 100%) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: white !important; }
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 8px !important;
}
.header-box {
    background: linear-gradient(135deg, #004c97 0%, #002d62 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.header-box h1 { font-size: 1.8rem; font-weight: 700; margin: 0; }
.header-box p  { opacity: 0.85; margin: 0.3rem 0 0; font-size: 0.95rem; }
.user-bubble {
    background: #004c97;
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0 0.4rem auto;
    max-width: 72%;
    width: fit-content;
    margin-left: auto;
    font-size: 0.97rem;
    line-height: 1.6;
}
.bot-bubble {
    background: white;
    color: #1a1a1a;
    padding: 1rem 1.3rem;
    border-radius: 18px 18px 18px 4px;
    border-left: 4px solid #ea7600;
    margin: 0.4rem 0;
    max-width: 82%;
    width: fit-content;
    font-size: 0.97rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,76,151,0.08);
}
.source-line {
    font-size: 0.8rem;
    color: #ea7600;
    margin-top: 0.6rem;
    padding-top: 0.6rem;
    border-top: 1px solid #f0f0f0;
}
.status-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.online  { background: #4ade80; }
.offline { background: #f87171; }
.stat-box {
    background: white;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #ea7600;
}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "messages"  : [],
    "connected" : False,
    "embeddings": None,
    "index"     : None,
    "llm"       : None,
    "tokens"    : 0,
    "msg_count" : 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── FREE models list for dropdown ─────────────────────────────────────────────
FREE_MODELS = {
    "⚡ LLaMA 3.3 70B (Best for RAG)"         : "meta-llama/llama-3.3-70b-instruct:free",
    "🌟 Gemini 2.0 Flash (1M context)"         : "google/gemini-2.0-flash-exp:free",
    "🧠 DeepSeek R1 (Best reasoning)"          : "deepseek/deepseek-r1:free",
    "🔥 Hermes 3 405B (Most powerful)"         : "nousresearch/hermes-3-llama-3.1-405b:free",
    "⚡ Mistral Small 3.1 (Fast)"              : "mistralai/mistral-small-3.1:free",
    "🤖 Gemma 3 27B (Google)"                  : "google/gemma-3-27b-it:free",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕉️ SSSIHL RAG")
    st.markdown("---")
    st.markdown("### ⚙️ API Settings")

    # Helper to load secrets safely whether on cloud or local
    def get_secret(key, default=""):
        try:
            return st.secrets.get(key, default)
        except Exception:
            return default

    openrouter_key = st.text_input(
        "🔑 OpenRouter API Key",
        type="password",
        value=get_secret("OPENROUTER_API_KEY", ""),
        placeholder="sk-or-v1-...",
        help="Get free key at openrouter.ai"
    )
    pinecone_key = st.text_input(
        "🔑 Pinecone API Key",
        type="password",
        value=get_secret("PINECONE_API_KEY", ""),
        placeholder="pcsk_..."
    )
    index_name = st.text_input("📌 Index Name", value=get_secret("PINECONE_INDEX", "saiinst"))

    st.markdown("### 🤖 Model Selection")
    model_label    = st.selectbox("Choose Model (All Free ✅)", list(FREE_MODELS.keys()))
    selected_model = FREE_MODELS[model_label]
    st.caption(f"`{selected_model}`")

    st.markdown("### 🎛️ Retrieval Settings")
    top_k     = st.slider("Chunks to retrieve", 3, 10, 5)
    min_score = st.slider("Min relevance score", 0.1, 0.9, 0.45, 0.05)

    connect = st.button("🚀 Connect & Start RAG", use_container_width=True)

    st.markdown("---")
    if st.session_state.connected:
        st.markdown("<span class='status-dot online'></span>**Connected**", unsafe_allow_html=True)
        st.caption(f"Model: `{selected_model.split('/') [0]}`")
    else:
        st.markdown("<span class='status-dot offline'></span>**Not connected**", unsafe_allow_html=True)

    st.markdown("### 📊 Session Stats")
    st.markdown(f"<div class='stat-box'>💬 Messages: <b>{st.session_state.msg_count}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stat-box'>🔢 Tokens: <b>{st.session_state.tokens}</b></div>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.msg_count = 0
        st.session_state.tokens    = 0
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; opacity:0.75; line-height:2;'>
    🌐 <a href='https://openrouter.ai' style='color:#ffae62;'>openrouter.ai</a> — 24+ free models<br>
    📦 Pinecone Vector DB<br>
    ⚡ FastEmbed BAAI BGE<br>
    🏛️ <a href='https://www.sssihl.edu.in' style='color:#ffae62;'>sssihl.edu.in</a>
    </div>
    """, unsafe_allow_html=True)

# ── Connect logic ─────────────────────────────────────────────────────────────
if connect:
    if not openrouter_key or not pinecone_key or not index_name:
        st.sidebar.error("⚠️ Fill all 3 fields!")
    else:
        with st.spinner("🔄 Loading models and connecting..."):
            try:
                # Embedding model (unchanged)
                st.session_state.embeddings = FastEmbedEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )
                # Pinecone (unchanged)
                pc = Pinecone(api_key=pinecone_key)
                st.session_state.index = pc.Index(index_name)

                # ✅ OpenRouter LLM — uses ChatOpenAI with custom base_url
                st.session_state.llm = ChatOpenAI(
                    model=selected_model,
                    openai_api_key=openrouter_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.2,
                    max_tokens=1024,
                    default_headers={
                        "HTTP-Referer": "https://sssihl.edu.in",
                        "X-Title"     : "SSSIHL Knowledge Assistant"
                    }
                )
                st.session_state.connected     = True
                st.session_state.active_model  = selected_model
                st.sidebar.success("✅ Connected!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant for Sri Sathya Sai Institute of Higher Learning (SSSIHL).
Answer ONLY using the context below. Be concise and cite source file and page number.
If not found say: "This information is not available in the provided documents."

History: {history}
Context: {context}
Question: {question}
Answer:
""")

# ── RAG functions ─────────────────────────────────────────────────────────────
def retrieve(query):
    vec     = st.session_state.embeddings.embed_query(query)
    results = st.session_state.index.query(vector=vec, top_k=top_k, include_metadata=True)
    parts, sources = [], []
    for m in results["matches"]:
        if m["score"] < min_score:
            continue
        text = m["metadata"].get("text", "")[:600]
        src  = m["metadata"].get("source_file", "doc")
        pg   = m["metadata"].get("page", "?")
        sc   = round(m["score"], 3)
        parts.append(f"[{src} | p.{pg} | {sc}]\\n{text}")
        sources.append(f"{src} p.{pg}")
    return "\\n\\n---\\n\\n".join(parts), list(set(sources))

def ask(question):
    vec     = st.session_state.embeddings.embed_query(question)
    results = st.session_state.index.query(vector=vec, top_k=1, include_metadata=True)
    if not results["matches"] or results["matches"][0]["score"] < min_score:
        return "⚠️ Question doesn't seem related to the documents. Please ask something relevant to SSSIHL.", []

    history = "\\n".join([
        f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in st.session_state.messages[-6:]
    ])
    context, sources = retrieve(f"{history}\\n{question}")
    if not context:
        return "⚠️ No relevant content found. Try rephrasing.", []

    response = st.session_state.llm.invoke(
        PROMPT.format_messages(history=history or "None", context=context, question=question)
    )
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        st.session_state.tokens += response.usage_metadata.get("total_tokens", 0)

    return response.content, sources

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-box'>
    <h1>🕉️ SSSIHL Knowledge Assistant</h1>
    <p>Sri Sathya Sai Institute of Higher Learning — A Modern Gurukula</p>
</div>
""", unsafe_allow_html=True)

# ── Not connected ─────────────────────────────────────────────────────────────
if not st.session_state.connected:
    st.info("👈 Enter your API keys in the sidebar and click **Connect & Start RAG**")
    st.markdown("""
    <div style='background:white; padding:1.5rem; border-radius:12px; border-left:4px solid #004c97;'>
    <b>How to get free API keys:</b><br><br>
    1️⃣ <b>OpenRouter</b> → <a href='https://openrouter.ai' target='_blank'>openrouter.ai</a> → Sign up → API Keys → Copy key (starts with <code>sk-or-v1-</code>)<br>
    2️⃣ <b>Pinecone</b> → <a href='https://app.pinecone.io' target='_blank'>app.pinecone.io</a> → Sign up → API Keys → Copy key<br>
    3️⃣ Enter your Pinecone index name (e.g. <code>saiinst</code>)<br>
    4️⃣ Pick any free model from the dropdown<br>
    5️⃣ Click <b>Connect & Start RAG</b> 🚀
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Suggestion chips ──────────────────────────────────────────────────────────
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
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>👤 &nbsp;{msg['content']}</div>", unsafe_allow_html=True)
    else:
        src_html = ""
        if msg.get("sources"):
            src_html = f"<div class='source-line'>📚 Sources: {' &nbsp;|&nbsp; '.join(msg['sources'])}</div>"
        st.markdown(f"<div class='bot-bubble'>🕉️ &nbsp;{msg['content']}{src_html}</div>", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about SSSIHL...")

if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.msg_count += 1
    with st.spinner("🧠 Searching documents..."):
        answer, sources = ask(question)
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
    st.rerun()
