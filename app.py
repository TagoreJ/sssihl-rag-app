import streamlit as st
import requests
import json
import os
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

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header & SEO ──────────────────────────────────────────────────────────────
st.markdown("""
<head>
    <meta name="description" content="Sia: The AI Knowledge Assistant for Sri Sathya Sai Institute of Higher Learning (SSSIHL). Ask questions about admissions, campus, and programs.">
    <meta name="keywords" content="SSSIHL, Sri Sathya Sai Institute of Higher Learning, Sia AI, SSSIHL AI, education chatbot, SSSIHL admissions">
    <meta name="author" content="SSSIHL">
</head>
<div class='header-box'>
    <h1>🎓 Sia — SSSIHL Knowledge Assistant</h1>
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
# Hardcoded optimal settings since the UI dropdowns were removed for simplicity
selected_model = list(FREE_MODELS.values())[0]  # Default to best model
top_k = 5
min_score = 0.45

with st.sidebar:
    st.markdown("## 🎓 Sia @ SSSIHL")
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
    st.markdown("### ℹ️ Limits & Info")
    st.markdown("<div style='font-size:0.8rem; opacity:0.8;'>This bot runs on free API tiers. If a model becomes overloaded, Sia will automatically switch to a backup model to keep answering you.</div>", unsafe_allow_html=True)
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
You are Sia, a smart, highly intelligent, and conversational AI assistant for Sri Sathya Sai Institute of Higher Learning (SSSIHL). 
You behave like a person (similar to ChatGPT) and take on the persona of a helpful student guide.

Here is some retrieved information from the institute's database:
---
{context}
---

INSTRUCTIONS:
1. Carefully read the user's question and THINK about what they are really asking.
2. Look at the retrieved information above. If the answer is in there, provide a clear, natural, and direct response.
3. If the retrieved information DOES NOT contain the answer to their specific question, DO NOT talk about unrelated topics from the context. Instead, politely admit that you don't have that exact knowledge in your database right now.
4. If the user just says a greeting (like "hi" or "how are you"), respond naturally without forcing facts into the conversation.
5. DO NOT cite file names, source paths, or page numbers in your text. Just answer seamlessly.
6. Do not constantly re-introduce yourself ("Hi, I'm Sia...") on every turn. Just answer the question like an ongoing chat.

History: 
{history}

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
        # Just store the raw text without citations so the AI doesn't read file paths it shouldn't say in the final chat.
        parts.append(text)
        sources.append(f"{src} p.{pg}")
    return "\n\n---\n\n".join(parts), list(set(sources))

def ask(question, model_to_use=None):
    history = "\n".join([
        f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}"
        for m in st.session_state.messages[-6:]
    ])
    
    # Retrieve using a combined search string
    context, sources = retrieve(f"{history}\n{question}")
    
    # Let Sia handle missing context conversationally
    if not context:
        context = "No relevant documents found. Please inform the user that you don't have this information in your database."

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
        # Show if a fallback model was used
        model_badge = ""
        if msg.get("model_used") and msg.get("model_used") != selected_model and msg.get("model_used") != "⚡ Sia's Local Memory":
            model_badge = f" <span style='font-size:0.7em; opacity:0.6; background:rgba(0,0,0,0.1); padding:2px 6px; border-radius:10px;'>{msg['model_used'].split('/')[-1]}</span>"
        elif msg.get("model_used") == "⚡ Sia's Local Memory":
            model_badge = f" <span style='font-size:0.7em; opacity:0.6; background:rgba(0,0,0,0.1); padding:2px 6px; border-radius:10px;'>⚡ Instant Memory</span>"
            
        st.markdown(f"<div class='bot-bubble'><b>Sia</b>{model_badge} &nbsp;{msg['content']}</div>", unsafe_allow_html=True)

# ── Local Persistent Cache (Memory Memory Graph Substitute) ────────────────
CACHE_FILE = "sia_memory_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_to_cache(question, answer, sources):
    cache = load_cache()
    # Store lowercased stripped version for fuzzy matching memory
    q_key = question.lower().strip()
    cache[q_key] = {"answer": answer, "sources": sources}
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask anything about SSSIHL...")

if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.msg_count += 1
    
    with st.spinner("🧠 Searching documents and generating response..."):
        # Check Local Cache first (Direct Answer Memory)
        local_cache = load_cache()
        q_key = question.lower().strip()
        
        if q_key in local_cache:
            # Found in persistent memory graph! Return instantly.
            answer = local_cache[q_key]["answer"]
            sources = local_cache[q_key]["sources"]
            model_used = "⚡ Sia's Local Memory"
        else:
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
                    
                    # Save successful exact new queries to persistent cache
                    save_to_cache(question, answer, sources)
                    
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
