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
    layout="centered",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero ── */
.hero {
    padding: 2.2rem 0 0.6rem 0;
    text-align: center;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7b9cff;
    margin-bottom: 1rem;
}
.hero-badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7b9cff;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.8px;
    background: linear-gradient(135deg, #ffffff 0%, #a0b4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #777;
    font-size: 0.88rem;
    margin: 0 auto;
    max-width: 480px;
    line-height: 1.6;
}
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.76rem;
    color: #555;
    padding: 1rem 0 1.2rem 0;
    justify-content: center;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
}

/* ── Chat meta ── */
.chat-meta {
    font-size: 0.74rem;
    color: #555;
    margin-top: 0.4rem;
    padding-left: 0.1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1c1c1c;
}
.sb-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #444;
    margin-bottom: 0.6rem;
    margin-top: 0.2rem;
}
.sb-desc {
    font-size: 0.85rem;
    color: #888;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}
.topic-chip {
    display: inline-block;
    background: #141414;
    border: 1px solid #242424;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #999;
    margin: 3px 3px;
}
.session-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #7b9cff;
    margin-bottom: 0.5rem;
}
.session-badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7b9cff;
    display: inline-block;
}
.session-id {
    font-family: monospace;
    color: #aaa;
    font-size: 0.82rem;
    text-transform: none;
    letter-spacing: 0;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    background: #0f0f0f !important;
    border: 1px solid #1e1e1e !important;
    color: #aaa !important;
    font-size: 0.82rem !important;
    padding: 0.35rem 0.7rem !important;
    border-radius: 6px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #3d3d3d !important;
    color: #fff !important;
    background: #161616 !important;
}

hr { border-color: #1a1a1a !important; }
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
# HERO
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        AI · Physics · B.Tech
    </div>
    <h1>{DOMAIN_NAME}</h1>
    <p>{DOMAIN_DESCRIPTION}</p>
</div>
<div class="status-bar">
    <span class="status-dot"></span>
    <span>{kb_count} chunks indexed &nbsp;·&nbsp; Groq LLaMA 3.3 · LangGraph · RAG</span>
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

    # ── Session badge at top ──
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
        <div class="session-badge">
            <span class="session-badge-dot"></span>
            Session &nbsp;<span class="session-id">{st.session_state.thread_id}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ New chat"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.rerun()

    st.divider()

    # ── About ──
    st.markdown('<div class="sb-label">About</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-desc">{DOMAIN_DESCRIPTION}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Topics ──
    st.markdown('<div class="sb-label">Topics Covered</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="topic-chip">{t}</span>' for t in KB_TOPICS)
    st.markdown(chips, unsafe_allow_html=True)

    st.divider()

    # ── Example questions ──
    st.markdown('<div class="sb-label">Try Asking</div>', unsafe_allow_html=True)
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