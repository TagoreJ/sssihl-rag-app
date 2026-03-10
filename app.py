import streamlit as st
import requests
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SSSIHL Knowledge Assistant",
    page_icon="🕉️",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-box'>
    <h1>🕉️ Sia — SSSIHL Knowledge Assistant</h1>
    <p>Hi, I am Sia! Your friendly guide to Sri Sathya Sai Institute of Higher Learning.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tokens" not in st.session_state:
    st.session_state.tokens = 0
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

# ── Load Secrets & Connect ────────────────────────────────────────────────────
@st.cache_resource
def init_rag():
    try:
        # Load from secrets directly (app will crash if not configured in Streamlit Cloud, which is desired)
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
        pinecone_key = st.secrets["PINECONE_API_KEY"]
        index_name = st.secrets.get("PINECONE_INDEX", "saiinst")
        
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        
        return embeddings, index, openrouter_key
    except Exception as e:
        st.error("⚠️ Error loading API keys. Make sure to configure Streamlit Cloud secrets with `OPENROUTER_API_KEY` and `PINECONE_API_KEY`.")
        st.stop()

# Auto-connects on page load using cached resource function
with st.spinner("🔄 Initializing system and connecting to DB..."):
    embeddings, index, openrouter_key = init_rag()

# ── Fetch FREE models list dynamically ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_free_models():
    default_models = {
        "⚡ LLaMA 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
        "🌟 Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",
        "🧠 DeepSeek R1": "deepseek/deepseek-r1:free",
    }
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            # Filter for models that have a pricing of 0 or 'free' in id
            free_models = {}
            for m in models_data:
                # OpenRouter usually tags free models with ':free' in ID
                if ":free" in m["id"] or "free" in m["id"].lower():
                    name = m.get("name", m["id"].split("/")[-1])
                    free_models[f"✨ {name}"] = m["id"]
            
            # If we successfully fetched free models, return them, else defaults
            if free_models:
                return free_models
    except Exception as e:
        pass
    return default_models

FREE_MODELS = get_free_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕉️ SSSIHL RAG")
    st.markdown("---")
    
    st.markdown("### 🤖 Model Selection")
    model_label    = st.selectbox("Choose Model (All Free ✅)", list(FREE_MODELS.keys()))
    selected_model = FREE_MODELS[model_label]
    st.caption(f"`{selected_model}`")

    st.markdown("### 🎛️ Retrieval Settings")
    top_k     = st.slider("Chunks to retrieve", 3, 10, 5)
    min_score = st.slider("Min relevance score", 0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    
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
    🌐 <a href='https://openrouter.ai' style='color:#ffae62;'>openrouter.ai</a><br>
    📦 Pinecone Vector DB<br>
    ⚡ FastEmbed BAAI BGE<br>
    🏛️ <a href='https://www.sssihl.edu.in' style='color:#ffae62;'>sssihl.edu.in</a>
    </div>
    """, unsafe_allow_html=True)

# ── Update the LLM with the active model dynamically ──────────────────────────
llm = ChatOpenAI(
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

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_template("""
You are Sia, a friendly, warm, and helpful assistant for Sri Sathya Sai Institute of Higher Learning (SSSIHL). 
You speak conversationally and normally to the user, like a helpful student guide. 
Answer their questions using the context below. Be concise but warm, and accurately cite the source file and page number.
If the information is not in the context, just politely let them know that you don't have that specific information right now.

History: {history}
Context: {context}
Question: {question}
Answer:
""")

# ── RAG functions ─────────────────────────────────────────────────────────────
def retrieve(query):
    vec     = embeddings.embed_query(query)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    parts, sources = [], []
    for m in results["matches"]:
        if m["score"] < min_score:
            continue
        text = m["metadata"].get("text", "")[:600]
        src  = m["metadata"].get("source_file", "doc")
        pg   = m["metadata"].get("page", "?")
        sc   = round(m["score"], 3)
        parts.append(f"[{src} | p.{pg} | {sc}]\n{text}")
        sources.append(f"{src} p.{pg}")
    return "\n\n---\n\n".join(parts), list(set(sources))

def ask(question, model_to_use=None):
    vec     = embeddings.embed_query(question)
    results = index.query(vector=vec, top_k=top_k, include_metadata=True)
    if not results["matches"] or results["matches"][0]["score"] < min_score:
        return "⚠️ Question doesn't seem related to the documents. Please ask something relevant to SSSIHL.", []

    history = "\n".join([
        f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in st.session_state.messages[-6:]
    ])
    context, sources = retrieve(f"{history}\n{question}")
    if not context:
        return "⚠️ No relevant content found. Try rephrasing.", []

    # Use the passed model, or the default LLM
    active_llm = llm
    if model_to_use:
        active_llm = ChatOpenAI(
            model=model_to_use,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=1024,
            default_headers={
                "HTTP-Referer": "https://sssihl.edu.in",
                "X-Title"     : "SSSIHL Knowledge Assistant"
            }
        )

    response = active_llm.invoke(
        PROMPT.format_messages(history=history or "None", context=context, question=question)
    )
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        st.session_state.tokens += response.usage_metadata.get("total_tokens", 0)

    return response.content, sources

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
        
        # Show if a fallback model was used
        model_badge = ""
        if msg.get("model_used") and msg.get("model_used") != selected_model:
            model_badge = f" <span style='font-size:0.7em; opacity:0.6; background:rgba(0,0,0,0.1); padding:2px 6px; border-radius:10px;'>Switched to config: {msg['model_used'].split('/')[-1]} due to load</span>"
            
        st.markdown(f"<div class='bot-bubble'>🕉️ Sia{model_badge} &nbsp;{msg['content']}{src_html}</div>", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about SSSIHL...")

if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.msg_count += 1
    
    with st.spinner("🧠 Searching documents and generating response..."):
        # Import exception for rate limit catching
        import openai
        
        success = False
        # Try the user's selected model first
        models_to_try = [selected_model] + [m for m in FREE_MODELS.values() if m != selected_model]
        
        answer = "⚠️ Sorry, all free models are currently overloaded. Please try again in a few minutes."
        sources = []
        model_used = selected_model
        
        for attempt_model in models_to_try:
            try:
                if attempt_model != selected_model:
                    st.toast(f"⚠️ Primary model rate limited. Trying fallback: {attempt_model.split('/')[-1]}", icon="🔄")
                    
                answer, sources = ask(question, model_to_use=attempt_model)
                success = True
                model_used = attempt_model
                break # Success! Break the loop
                
            except openai.RateLimitError:
                continue # Try the next model
            except Exception as e:
                # If it's another kind of error, log it and still try next
                print(f"Error with model {attempt_model}: {e}")
                continue
                
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources,
        "model_used": model_used
    })
    st.rerun()
