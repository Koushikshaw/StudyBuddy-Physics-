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

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DOMAIN_NAME,
    page_icon="⚛️",
    layout="centered",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit header decoration */
.stAppHeader { background: transparent !important; }

/* Hero header */
.hero {
    padding: 2rem 0 1.2rem 0;
    text-align: center;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #888;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

/* Status bar */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    color: #666;
    padding: 0.4rem 0 1rem 0;
    justify-content: center;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
}

/* Chat meta line (faithfulness, route, sources) */
.chat-meta {
    font-size: 0.72rem;
    color: #666;
    margin-top: 0.35rem;
    padding-left: 0.1rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #1e1e1e;
}
.sidebar-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #555;
    margin-bottom: 0.5rem;
}
.topic-chip {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #aaa;
    margin: 2px 2px;
}

/* Example question buttons */
div[data-testid="stButton"] button {
    background: #111 !important;
    border: 1px solid #222 !important;
    color: #bbb !important;
    font-size: 0.78rem !important;
    padding: 0.3rem 0.6rem !important;
    border-radius: 6px !important;
    text-align: left !important;
    width: 100% !important;
    transition: border-color 0.2s, color 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #444 !important;
    color: #fff !important;
}

/* Chat input */
div[data-testid="stChatInput"] {
    border-top: 1px solid #1e1e1e;
    padding-top: 0.5rem;
}

/* Divider */
hr { border-color: #1e1e1e !important; }
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
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>⚛️ {DOMAIN_NAME}</h1>
    <p>{DOMAIN_DESCRIPTION}</p>
</div>
<div class="status-bar">
    <span class="status-dot"></span>
    <span>{kb_count} chunks indexed &nbsp;·&nbsp; Powered by Groq + LangGraph</span>
</div>
""", unsafe_allow_html=True)

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
    st.markdown('<div class="sidebar-title">Physics Study Buddy</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.78rem;color:#666;margin-bottom:1rem;">{DOMAIN_DESCRIPTION}</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-title">Topics Covered</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="topic-chip">{t}</span>' for t in KB_TOPICS)
    st.markdown(chips, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-title">Try Asking</div>', unsafe_allow_html=True)
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

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div style="font-size:0.7rem;color:#555;">Session</div>'
                    f'<div style="font-size:0.78rem;color:#888;font-family:monospace;">{st.session_state.thread_id}</div>',
                    unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ New chat"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.rerun()

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
        with st.spinner("🔭 Thinking…"):
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