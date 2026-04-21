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
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #1a1a1a;
    color: #e0e0e0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #222222 !important;
    border-right: 1px solid #2e2e2e !important;
    padding-top: 1.2rem;
}
section[data-testid="stSidebar"] > div {
    padding: 1rem 1.2rem;
}

.sb-section-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 0.55rem;
}
.sb-section-title .icon {
    font-size: 1rem;
}
.sb-desc {
    font-size: 0.83rem;
    color: #aaaaaa;
    line-height: 1.65;
    margin-bottom: 0.2rem;
}
.sb-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: #666;
    margin-bottom: 0.5rem;
    margin-top: 0.2rem;
}
.session-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.session-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: #cccccc;
}
.session-id {
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #4fc3f7;
    background: #1a2a35;
    border-radius: 4px;
    padding: 1px 7px;
}
.topic-item {
    font-size: 0.83rem;
    color: #aaaaaa;
    padding: 2px 0 2px 0.3rem;
    line-height: 1.7;
}
.topic-item::before {
    content: "• ";
    color: #555;
}

/* ── Main area ── */
.main-wrap {
    max-width: 820px;
    margin: 0 auto;
    padding: 2rem 1rem 1rem;
}

/* ── Page title ── */
.page-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f0f0;
    margin-bottom: 0.45rem;
    letter-spacing: -0.5px;
}
.page-title .icon {
    font-size: 1.9rem;
}
.page-subtitle {
    font-size: 0.88rem;
    color: #888;
    line-height: 1.6;
    margin-bottom: 1.2rem;
    max-width: 640px;
}

/* ── KB status banner ── */
.kb-banner {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: #1a3a25;
    border: 1px solid #2a5a35;
    border-radius: 8px;
    padding: 0.75rem 1.1rem;
    font-size: 0.86rem;
    color: #4ade80;
    margin-bottom: 1.6rem;
    font-weight: 500;
}
.kb-banner .check {
    font-size: 1rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #252525 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"] p {
    font-size: 0.91rem !important;
    line-height: 1.7 !important;
    color: #d8d8d8 !important;
}

/* ── Chat meta line ── */
.chat-meta {
    font-size: 0.74rem;
    color: #555;
    margin-top: 0.4rem;
    padding-left: 0.1rem;
}

/* ── Chat input ── */
[data-testid="stBottom"] {
    background: #1a1a1a !important;
    border-top: 1px solid #2e2e2e !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] {
    background: #252525 !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: #252525 !important;
    border: none !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4fc3f7 !important;
    box-shadow: 0 0 0 3px rgba(79,195,247,0.08) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #555 !important;
}
[data-testid="stChatInput"] button {
    background: #4fc3f7 !important;
    border-radius: 8px !important;
    color: #0d0d0d !important;
    border: none !important;
    margin: 4px !important;
}

/* ── Sidebar buttons ── */
div[data-testid="stButton"] button {
    background: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    color: #999 !important;
    font-size: 0.82rem !important;
    padding: 0.35rem 0.7rem !important;
    border-radius: 6px !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #444 !important;
    color: #fff !important;
    background: #2a2a2a !important;
}

hr { border-color: #2a2a2a !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #666 !important;
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

    # About
    st.markdown(f"""
    <div class="sb-section-title"><span class="icon">ℹ️</span> About</div>
    <div class="sb-desc">{DOMAIN_DESCRIPTION}</div>
    """, unsafe_allow_html=True)

    st.divider()

    # Session ID
    st.markdown(f"""
    <div class="session-row">
        <span class="session-label">Session ID:</span>
        <span class="session-id">{st.session_state.thread_id}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()

    # Topics
    st.markdown('<div class="sb-section-title"><span class="icon">🗂️</span> Topics covered:</div>', unsafe_allow_html=True)
    topics_html = "".join(f'<div class="topic-item">{t.replace("_", " ")}</div>' for t in KB_TOPICS)
    st.markdown(topics_html, unsafe_allow_html=True)

    st.divider()

    # Example questions
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
# MAIN AREA
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-wrap">
    <div class="page-title">
        <span class="icon">⚛️</span>{DOMAIN_NAME}
    </div>
    <div class="page-subtitle">{DOMAIN_DESCRIPTION}</div>
    <div class="kb-banner">
        <span class="check">✅</span>
        Knowledge base loaded — {kb_count} chunks indexed
    </div>
</div>
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