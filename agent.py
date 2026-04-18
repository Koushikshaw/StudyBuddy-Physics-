"""
agent.py — Physics Study Buddy
Node names all end with _node to match graph registration.
Uses chromadb DefaultEmbeddingFunction instead of sentence_transformers.
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
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

load_dotenv()

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
PDF_FOLDER = "pdfs"

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
    question: str
    messages: List[dict]
    route: str          # "retrieve" | "memory_only" | "tool" | "chat"
    intent: str         # "calculator"|"convert"|"solve"|"compare"|"plan"|"simplify"|"search"
    retrieved: str
    sources: List[str]
    tool_name: str
    tool_input: str
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    formula_used: str
    calculation_steps: str
    units: str
    difficulty_level: str
    search_results: str


# ─────────────────────────────────────────────────────────
# MODEL + KB LOADER
# ─────────────────────────────────────────────────────────
def load_llm_and_kb():
    """Returns (llm, embedder, chroma_collection)."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = DefaultEmbeddingFunction()

    client = chromadb.Client()
    try:
        client.delete_collection("physics_kb")
    except Exception:
        pass
    collection = client.create_collection(
        "physics_kb",
        embedding_function=embedder,
    )

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
        ids=[f"id_{i}" for i in range(len(texts))],
        metadatas=metas,
    )

    return llm, embedder, collection


# ─────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────
def make_nodes(llm, embedder, collection):
    """Factory returning node functions. All dict keys end with _node."""

    # ── Node 1: memory_node ───────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {
            "messages": msgs,
            "answer": "",
            "retrieved": "",
            "sources": [],
            "tool_name": "",
            "tool_result": "",
            "route": "",
            "intent": "",
            "faithfulness": 0.0,
            "eval_retries": 0,
        }

    # ── Node 2: router_node ───────────────────────────────
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
- chat        → conversational message (greetings, thanks, "ok", "good", acknowledgements)

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

Reply with ONLY ONE WORD: retrieve OR memory_only OR tool OR chat
"""
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        elif "chat" in decision:
            decision = "chat"
        else:
            decision = "retrieve"
        return {"route": decision}

    # ── Node 3: intent_classifier_node ───────────────────
    def intent_classifier_node(state: CapstoneState) -> dict:
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

    # ── Node 4: retrieval_node ────────────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        results = collection.query(query_texts=[state["question"]], n_results=3)
        chunks = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        if not chunks:
            return {"retrieved": "", "sources": []}
        topics = [m.get("topic", "Unknown") for m in metas]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    # ── Node 5: skip_retrieval_node ───────────────────────
    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    # ── Node 6: tool_node ─────────────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        intent   = state.get("intent", "").lower()
        tool_name   = ""
        tool_result = ""

        if intent == "search":
            tool_name = "web_search"
            try:
                from duckduckgo_search import DDGS
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

    # ── Node 7: answer_node ───────────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        route        = state.get("route", "retrieve")

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        chat_note = ""
        if route == "chat":
            chat_note = "\n- If the message is conversational (greeting, thanks, acknowledgement), reply briefly and naturally in 1 sentence."

        retry_note = ""
        if eval_retries > 0:
            retry_note = (
                "\n\nIMPORTANT: Your previous answer was not sufficiently grounded. "
                "Strictly ensure every statement comes from the context. No hallucination."
            )

        if context:
            system_content = f"""You are a Physics Study Buddy for B.Tech students.

Your goal is to explain physics concepts clearly, accurately, and in a structured way.

STRICT RULES:
- Answer ONLY using the provided context (knowledge base or tool result)
- DO NOT use outside knowledge
- If the answer is not in the context, say: "I don't have that information in my knowledge base."
- Do NOT hallucinate — be precise and educational{chat_note}

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

{context}{retry_note}"""
        else:
            system_content = (
                "You are a Physics Study Buddy.\n\n"
                "Answer using the conversation history only.\n"
                f"If unsure, say: \"I don't have that information in my knowledge base.\"{chat_note}"
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

    # ── Node 8: eval_node ─────────────────────────────────
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

    # ── Node 9: save_node ─────────────────────────────────
    def save_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "assistant", "content": state["answer"]}]
        if len(msgs) > 8:
            msgs = msgs[-8:]
        return {"messages": msgs}

    return {
        "memory_node":            memory_node,
        "router_node":            router_node,
        "intent_classifier_node": intent_classifier_node,
        "retrieval_node":         retrieval_node,
        "skip_retrieval_node":    skip_retrieval_node,
        "tool_node":              tool_node,
        "answer_node":            answer_node,
        "eval_node":              eval_node,
        "save_node":              save_node,
    }


# ─────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────
def build_agent(llm, embedder, collection):
    nodes   = make_nodes(llm, embedder, collection)
    memory  = MemorySaver()
    builder = StateGraph(CapstoneState)

    builder.add_node("memory_node",            nodes["memory_node"])
    builder.add_node("router_node",            nodes["router_node"])
    builder.add_node("intent_classifier_node", nodes["intent_classifier_node"])
    builder.add_node("retrieval_node",         nodes["retrieval_node"])
    builder.add_node("skip_retrieval_node",    nodes["skip_retrieval_node"])
    builder.add_node("tool_node",              nodes["tool_node"])
    builder.add_node("answer_node",            nodes["answer_node"])
    builder.add_node("eval_node",              nodes["eval_node"])
    builder.add_node("save_node",              nodes["save_node"])

    builder.set_entry_point("memory_node")
    builder.add_edge("memory_node", "router_node")

    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":
            return "intent_classifier_node"
        elif r in ("memory_only", "chat"):
            return "skip_retrieval_node"
        return "retrieval_node"

    builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "intent_classifier_node": "intent_classifier_node",
            "retrieval_node":         "retrieval_node",
            "skip_retrieval_node":    "skip_retrieval_node",
        },
    )

    builder.add_edge("intent_classifier_node", "tool_node")
    builder.add_edge("tool_node",              "answer_node")
    builder.add_edge("retrieval_node",         "answer_node")
    builder.add_edge("skip_retrieval_node",    "answer_node")
    builder.add_edge("answer_node",            "eval_node")

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save_node"
        return "answer_node"

    builder.add_conditional_edges(
        "eval_node",
        eval_decision,
        {"answer_node": "answer_node", "save_node": "save_node"},
    )

    builder.add_edge("save_node", END)
    return builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────
# CONVENIENCE: ask()
# ─────────────────────────────────────────────────────────
def ask(agent_app, question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return agent_app.invoke({"question": question}, config=config)