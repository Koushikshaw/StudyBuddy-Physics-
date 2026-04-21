"""
capstone_streamlit.py — Physics Study Buddy
"""

import streamlit as st
import uuid

from agent import (
    DOMAIN_NAME,
    DOMAIN_DESCRIPTION,
    KB_TOPICS,
    load_llm_and_kb,
    build_agent,
    ask,
)

st.set_page_config(
    page_title=DOMAIN_NAME,
    page_icon="⚛️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS  —  matches the Agentic AI screenshot style
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #0e1117 !important;
    color: #e0e0e0 !important;
}
[data-testid="stAppViewContainer"] {
    background: #0e1117 !important;
}
[data-testid="stMain"] {
    background: #0e1117 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a1c23 !important;
    border-right: 1px solid #2a2d36 !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.2rem 1rem !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    font-size: 0.85rem !important;
    color: #b0b8c8 !important;
    line-height: 1.65 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #e8eaf0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2a2d36 !important;
    margin: 0.75rem 0 !important;
}

/* ── Sidebar buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    color: #9099aa !important;
    font-size: 0.84rem !important;
    padding: 0.3rem 0 !important;
    text-align: left !important;
    width: 100% !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    color: #ffffff !important;
    background: transparent !important;
}

/* ── Main content width ── */
[data-testid="stMainBlockContainer"] {
    max-width: 860px !important;
    padding: 2rem 2rem 1rem !important;
    margin: 0 auto !important;
}

/* ── Page title ── */
.page-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f2f6;
    margin-bottom: 0.5rem;
    letter-spacing: -0.3px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #7a8394;
    line-height: 1.65;
    margin-bottom: 1.2rem;
    max-width: 680px;
}

/* ── KB banner ── */
.kb-banner {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    background: #1a3a25;
    border: 1px solid #2d5c3a;
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    font-size: 0.87rem;
    color: #4ade80;
    font-weight: 500;
    margin-bottom: 1.4rem;
}

/* ── Chat messages — no card borders, clean bubbles ── */
[data-testid="stChatMessage"] {
    background: #181b22 !important;
    border: 1px solid #22262f !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"] p {
    font-size: 0.92rem !important;
    line-height: 1.72 !important;
    color: #d8dce8 !important;
    margin: 0 !important;
}

/* ── Chat meta (route / sources) ── */
.chat-meta {
    font-size: 0.75rem;
    color: #4a5260;
    margin-top: 0.45rem;
}

/* ── Chat input container ── */
[data-testid="stBottom"] {
    background: #0e1117 !important;
    border-top: none !important;
    padding: 0.5rem 0 1rem !important;
}
[data-testid="stChatInput"] {
    background: #181b22 !important;
    border: 1px solid #2a2d36 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #e05252 !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #d8dce8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.8rem 1rem !important;
    caret-color: #e05252 !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #4a5260 !important;
}
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    background: #3b82f6 !important;
    border-radius: 7px !important;
    color: #fff !important;
    border: none !important;
    margin: 5px !important;
    transition: background 0.15s !important;
}
[data-testid="stChatInput"] button:hover {
    background: #2563eb !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #4a5260 !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD AGENT
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading knowledge base…")
def get_agent():
    llm, embedder, collection = load_llm_and_kb()
    agent_app = build_agent(llm, embedder, collection)
    return agent_app, collection

try:
    agent_app, collection = get_agent()
    kb_count = collection.count()
except Exception as e:
    st.error(f"❌ Failed to load agent: {e}")
    st.info(
        "**Checklist:**\n"
        "1. Make sure a `pdfs/` folder exists with your physics PDFs.\n"
        "2. Set `GROQ_API_KEY` in Streamlit Cloud → App Settings → Secrets."
    )
    st.stop()

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(DOMAIN_DESCRIPTION)
    st.divider()

    st.markdown(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()

    st.markdown("### 🗂️ Topics covered:")
    for topic in KB_TOPICS:
        st.write(f"• {topic.replace('_', ' ')}")
    st.divider()

    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.divider()

    st.markdown("**💡 Try asking:**")
    example_questions = [
        "What is Simple Harmonic Motion?",
        "Explain damped harmonic motion",
        "What are Maxwell's equations?",
        "Compare interference vs diffraction",
        "Create a study plan for wave motion",
        "Explain laser components in simple terms",
        "What is Fraunhofer diffraction?",
    ]
    for eq in example_questions:
        if st.button(eq, key=eq):
            st.session_state._inject_question = eq
            st.rerun()

# ─────────────────────────────────────────────────────────
# MAIN AREA — title + KB banner
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-title">⚛️ {DOMAIN_NAME}</div>
<div class="page-subtitle">{DOMAIN_DESCRIPTION}</div>
<div class="kb-banner">✅ &nbsp;Knowledge base loaded — {kb_count} chunks indexed</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            parts = []
            if meta.get("faithfulness"):
                parts.append(f"faithfulness {meta['faithfulness']:.2f}")
            if meta.get("route"):
                parts.append(f"route: {meta['route']}")
            if meta.get("sources"):
                parts.append(f"sources: {', '.join(set(meta['sources']))}")
            if parts:
                st.markdown(
                    f'<div class="chat-meta">{" · ".join(parts)}</div>',
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────
injected   = st.session_state.pop("_inject_question", None)
user_input = st.chat_input("Ask a physics question…") or injected

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = ask(agent_app, user_input, thread_id=st.session_state.thread_id)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        faith   = result.get("faithfulness", 0.0)
        route   = result.get("route", "")
        sources = result.get("sources", [])

        parts = []
        if faith:
            parts.append(f"faithfulness {faith:.2f}")
        if route:
            parts.append(f"route: {route}")
        if sources:
            parts.append(f"sources: {', '.join(set(sources))}")
        if parts:
            st.markdown(
                f'<div class="chat-meta">{" · ".join(parts)}</div>',
                unsafe_allow_html=True
            )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "meta": {
            "faithfulness": faith,
            "route":        route,
            "sources":      sources,
        },
    })