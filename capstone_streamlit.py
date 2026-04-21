"""
capstone_streamlit.py — Study Buddy (Physics)
Run: streamlit run capstone_streamlit.py
"""
import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
DOMAIN_NAME = "Study Buddy (Physics)"
DOMAIN_DESCRIPTION = "AI assistant for physics concepts, numericals, derivations, and study plans."
KB_TOPICS = [
    "Simple_Harmonic_Motion",
    "Damped_Harmonic_Motion",
    "Laws_of_Motion",
    "Work_Energy_Power",
    "Waves",
    "Oscillations",
    "Thermodynamics",
    "Optics",
    "Modern_Physics"
]
st.set_page_config(page_title=DOMAIN_NAME, page_icon="📘")

# ── DESIGN ONLY: inject custom CSS + header ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=Lora:ital,wght@0,400;1,400&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #13161e;
    --border:  #1f2435;
    --accent:  #e8c547;
    --accent2: #4fc3f7;
    --text:    #dce3f0;
    --muted:   #6b7594;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Lora', Georgia, serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stAppViewContainer"] > .main {
    max-width: 780px;
    margin: 0 auto;
    padding: 0 1.5rem;
}

.site-header {
    border-bottom: 1px solid var(--border);
    padding: 2.2rem 0 1.6rem;
    margin-bottom: 2rem;
}
.site-header .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.site-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem;
    line-height: 1.1;
}
.site-header .sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
}
.accent-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 0.5rem;
    position: relative; top: -1px;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.4rem !important;
}
[data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
    border-top: 1px solid var(--border);
    padding-top: 1.4rem !important;
}

[data-testid="chatAvatarIcon-user"] svg,
[data-testid="chatAvatarIcon-assistant"] svg { display: none; }

[data-testid="chatAvatarIcon-user"]::before {
    content: "YOU";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.1em;
    color: var(--accent);
    background: #1e1f0f;
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 3px 5px;
    display: flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; box-sizing: border-box;
}
[data-testid="chatAvatarIcon-assistant"]::before {
    content: "PHY";
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.1em;
    color: var(--accent2);
    background: #0c1820;
    border: 1px solid var(--accent2);
    border-radius: 4px;
    padding: 3px 5px;
    display: flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; box-sizing: border-box;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.97rem !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
}
[data-testid="stChatMessage"] code {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85em !important;
    background: #1a1e2e !important;
    color: var(--accent) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stChatInput"] {
    border-top: 1px solid var(--border) !important;
    padding-top: 1rem !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    caret-color: var(--accent) !important;
    padding: 0.85rem 1rem !important;
    resize: none !important;
    transition: border-color 0.2s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(232,197,71,0.08) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 6px !important;
    color: #0d0f14 !important;
    border: none !important;
}

[data-testid="stSpinner"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>

<div class="site-header">
    <div class="label"><span class="accent-dot"></span>Knowledge Assistant</div>
    <h1>Physics Study Buddy</h1>
    <div class="sub">SHM · Thermodynamics · Optics · Waves · Modern Physics</div>
</div>
""", unsafe_allow_html=True)

# ── Everything below is UNCHANGED from original ──────────────────────────────
@st.cache_resource
def load_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except:
        pass
    collection = client.create_collection("capstone_kb")
    PDF_FOLDER = "pdfs"
    all_docs = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
            docs = loader.load()
            for d in docs:
                d.metadata["topic"] = file.replace(".pdf", "")
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[f"id_{i}" for i in range(len(texts))],
        metadatas=metas
    )
    return collection, embedder, llm
collection, embedder, llm = load_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
if prompt := st.chat_input("Ask a physics question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = collection.query(
                query_embeddings=embedder.encode([prompt]).tolist(),
                n_results=3
            )
            context = "\n\n".join(results["documents"][0])
            system_prompt = f"""
You are a physics assistant. Answer using ONLY this context:
{context}
"""
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            answer = response.content
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})