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
st.title(f"📘 {DOMAIN_NAME}")
st.caption(DOMAIN_DESCRIPTION)

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