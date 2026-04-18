"""
agent.py — Shared agent module for Physics Study Buddy
This file contains the CapstoneState, all node functions,
and the graph assembly. Imported by capstone_streamlit.py.
"""

import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
PDF_FOLDER = "pdfs"   # folder containing your physics PDF files

DOMAIN_NAME = "Physics Study Buddy"
DOMAIN_DESCRIPTION = (
    "An AI-powered Study Buddy for B.Tech students to understand Physics concepts "
    "using a knowledge base, memory, and intelligent tools like step-by-step solving, "
    "unit conversion, concept comparison, and study planning."
)

KB_TOPICS = [
    "Simple Harmonic Motion",
    "Damped Harmonic Motion",
    "EM Waves Transverse Nature",
    "Fraunhofer Diffraction",
    "Inference of Light",
    "Laser Components",
    "Maxwells Equations",
    "Quantum Process",
    "Waves and Wave Motion",
]

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


# ─────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    # Input
    question: str
    # Memory
    messages: List[dict]
    # Routing
    route: str          # "retrieve" | "memory_only" | "tool"
    intent: str         # "calculator"|"convert"|"solve"|"compare"|"plan"|"simplify"|"search"
    # RAG
    retrieved: str
    sources: List[str]
    # Tool
    tool_name: str
    tool_input: str
    tool_result: str
    # Answer
    answer: str
    # Quality control
    faithfulness: float
    eval_retries: int
    # Physics-specific
    formula_used: str
    calculation_steps: str
    units: str
    difficulty_level: str
    search_results: str


# ─────────────────────────────────────────────────────────
# MODEL + KB LOADER  (called once by Streamlit's @st.cache_resource)
# ─────────────────────────────────────────────────────────
def load_llm_and_kb():
    """
    Returns (llm, embedder, chroma_collection).
    Loads PDFs from the `pdfs/` folder, splits, embeds, and
    stores in a fresh in-memory ChromaDB collection.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass
    collection = client.create_collection("capstone_kb")

    all_docs = []
    if os.path.isdir(PDF_FOLDER):
        for fname in os.listdir(PDF_FOLDER):
            if fname.endswith(".pdf"):
                path = os.path.join(PDF_FOLDER, fname)
                loader = PyPDFLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["topic"] = fname.replace(".pdf", "")
                all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError(
            f"No PDFs found in '{PDF_FOLDER}/' folder. "
            "Please create a 'pdfs/' directory next to this file "
            "and add your physics PDF files there."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[f"id_{i}" for i in range(len(texts))],
        metadatas=metas,
    )

    return llm, embedder, collection


# ─────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────
def make_nodes(llm, embedder, collection):
    """
    Factory that returns node functions bound to llm/embedder/collection.
    """

    # ── Node 1: Memory ────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    # ── Node 2: Router ────────────────────────────────────
    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(
            f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
        ) or "none"

        prompt = f"""
You are a routing assistant for a Physics Study Buddy chatbot (B.Tech level).

Decide how to answer the user's question.

Options:
- retrieve    → physics concepts, definitions, derivations, formulas from knowledge base
- memory_only → question refers to previous conversation (e.g. "what did you just say?")
- tool        → computation, comparison, or external info needed

IMPORTANT: If question has a false assumption, choose "retrieve" to correct it.

Use tool when:
- numerical problem      → calculator/solver
- unit conversion        → converter
- comparison             → compare tool
- study plan             → plan tool
- simple explanation     → simplify tool
- out-of-syllabus info   → web search

Recent conversation: {recent}
Current question: {question}

Reply with ONLY ONE WORD: retrieve OR memory_only OR tool
"""
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}

    # ── Node 3a: Intent classifier ─────────────────────────
    def intent_node(state: CapstoneState) -> dict:
        question = state["question"]
        prompt = f"""
Classify the intent of this physics question into ONE word:
- calculator → numerical physics problem to solve
- convert    → unit conversion needed
- solve      → step-by-step derivation or problem solving
- compare    → comparing two or more physics concepts
- plan       → study plan or learning roadmap
- simplify   → explain in simple terms or with analogy
- search     → out-of-syllabus or latest information

Question: {question}

Reply with ONLY one word: calculator, convert, solve, compare, plan, simplify, or search
"""
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
        valid = ("calculator", "convert", "solve", "compare", "plan", "simplify", "search")
        if intent not in valid:
            intent = "search"
        return {"intent": intent}

    # ── Node 3b: Retrieval ─────────────────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        if not chunks:
            return {"retrieved": "", "sources": []}
        topics = [m.get("topic", "Unknown") for m in metas]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    # ── Node 4: Tool ──────────────────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        intent   = state.get("intent", "").lower()
        tool_name   = ""
        tool_result = ""

        if intent == "search":
            tool_name = "web_search"
            try:
                from ddgs import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(question, max_results=3))
                tool_result = "\n".join(
                    f"{r['title']}: {r['body'][:200]}" for r in results
                )
            except Exception as e:
                tool_result = f"Web search error: {e}"

        elif intent == "calculator":
            tool_name = "calculator"
            try:
                expr = question.replace("^", "**")
                result = eval(expr, {"__builtins__": {}})
                tool_result = f"Final Answer: {result}"
            except Exception:
                tool_result = (
                    f"Solve this numerical physics problem step-by-step:\n\n{question}\n\n"
                    "Include:\n- Given values\n- Formula used\n- Substitution\n- Final answer with units"
                )

        elif intent == "convert":
            tool_name = "unit_converter"
            tool_result = (
                f"Convert the following units:\n\n{question}\n\n"
                "Provide:\n- Conversion factor\n- Conversion steps\n- Final converted value with units"
            )

        elif intent == "solve":
            tool_name = "step_solver"
            tool_result = (
                f"Solve/explain this physics problem step-by-step:\n\n{question}\n\n"
                "Include:\n- Concept used\n- Equations / derivation steps\n"
                "- Substitution\n- Final conclusion"
            )

        elif intent == "plan":
            tool_name = "study_plan"
            tool_result = (
                f"Create a structured physics study plan for:\n\n{question}\n\n"
                "Include:\n- Key topics\n- Order of study\n- Key formulas\n"
                "- Practice strategy\n- Timeline"
            )

        elif intent == "compare":
            tool_name = "compare"
            tool_result = (
                f"Compare the following physics concepts:\n\n{question}\n\n"
                "Include:\n- Definitions\n- Key differences\n- Equations\n- Examples"
            )

        elif intent == "simplify":
            tool_name = "simplifier"
            tool_result = (
                f"Explain this in very simple terms:\n\n{question}\n\n"
                "Use:\n- Easy language\n- Analogies\n- Intuition (no jargon)"
            )

        else:
            tool_name   = "none"
            tool_result = ""

        return {"tool_name": tool_name, "tool_result": tool_result}

    # ── Node 5: Answer ────────────────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are a Physics Study Buddy for B.Tech students.

Your goal is to explain physics concepts clearly, accurately, and in a structured way.

STRICT RULES:
- Answer ONLY using the provided context (knowledge base or tool result)
- DO NOT use outside knowledge
- If the answer is not in the context, say: "I don't have that information in my knowledge base."
- Do NOT hallucinate — be precise and educational

FORMAT YOUR ANSWER LIKE THIS (when applicable):

1. Definition:
   - Clear explanation of the concept

2. Key Concept / Explanation:
   - Intuition behind the concept
   - Important points

3. Formula / Equation (if present):
   - Write the formula clearly
   - Explain each variable

4. Step-by-Step Explanation (for problems/derivations):
   - Given → Formula used → Substitution → Final result

5. Example (if possible):
   - Simple real-world or textbook example

Use bullet points and clean formatting.

{context}"""
        else:
            system_content = (
                "You are a Physics Study Buddy.\n\n"
                "Answer using the conversation history only.\n"
                "If unsure, say: \"I don't have that information in my knowledge base.\""
            )

        if eval_retries > 0:
            system_content += (
                "\n\nIMPORTANT: Your previous answer was not sufficiently grounded. "
                "Strictly ensure every statement comes from the context. No hallucination."
            )

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    # ── Node 6: Evaluator ─────────────────────────────────
    def eval_node(state: CapstoneState) -> dict:
        answer  = state.get("answer", "")
        context = (state.get("retrieved", "") + "\n" + state.get("tool_result", ""))[:500]
        retries = state.get("eval_retries", 0)

        if not context.strip():
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = (
            "Rate faithfulness: does this answer use ONLY information from the context?\n"
            "Reply with ONLY a decimal number between 0.0 and 1.0.\n\n"
            f"Context: {context}\nAnswer: {answer[:300]}"
        )
        try:
            score = float(
                llm.invoke(prompt).content.strip().split()[0].replace(",", ".")
            )
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    # ── Node 7: Save memory ───────────────────────────────
    def save_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "assistant", "content": state["answer"]}]
        if len(msgs) > 8:
            msgs = msgs[-8:]
        return {"messages": msgs}

    return {
        "memory":         memory_node,
        "router":         router_node,
        "intent":         intent_node,
        "retrieval":      retrieval_node,
        "skip_retrieval": skip_retrieval_node,
        "tool":           tool_node,
        "answer":         answer_node,
        "eval":           eval_node,
        "save":           save_node,
    }


# ─────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────
def build_agent(llm, embedder, collection):
    """Builds and compiles the LangGraph StateGraph. Returns compiled app."""
    nodes  = make_nodes(llm, embedder, collection)
    memory = MemorySaver()
    builder = StateGraph(CapstoneState)

    builder.add_node("memory",      nodes["memory"])
    builder.add_node("router",      nodes["router"])
    builder.add_node("intent",      nodes["intent"])
    builder.add_node("retrieve",    nodes["retrieval"])
    builder.add_node("memory_only", nodes["skip_retrieval"])
    builder.add_node("tool",        nodes["tool"])
    builder.add_node("answer",      nodes["answer"])
    builder.add_node("eval",        nodes["eval"])
    builder.add_node("save",        nodes["save"])

    builder.set_entry_point("memory")
    builder.add_edge("memory", "router")

    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":
            return "intent"
        elif r == "memory_only":
            return "memory_only"
        return "retrieve"

    builder.add_conditional_edges(
        "router", route_decision,
        {"intent": "intent", "retrieve": "retrieve", "memory_only": "memory_only"},
    )

    builder.add_edge("intent",      "tool")
    builder.add_edge("tool",        "answer")
    builder.add_edge("retrieve",    "answer")
    builder.add_edge("memory_only", "answer")
    builder.add_edge("answer",      "eval")

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    builder.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"},
    )

    builder.add_edge("save", END)
    return builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────
# CONVENIENCE: ask() for notebook / testing
# ─────────────────────────────────────────────────────────
def ask(agent_app, question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return agent_app.invoke({"question": question}, config=config)