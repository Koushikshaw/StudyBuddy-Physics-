# ⚛️ StudyBuddy — Physics AI Tutor for B.Tech

> An Agentic RAG-powered AI chatbot that helps B.Tech students understand Physics concepts using a local knowledge base, intelligent routing, multi-tool support, and memory — all running in a clean Streamlit UI.

---

## 🧠 What is StudyBuddy?

**StudyBuddy** is an AI-powered study assistant built specifically for B.Tech Physics students. It combines:

- 📚 **RAG (Retrieval-Augmented Generation)** — answers grounded in your actual physics PDFs
- 🤖 **Agentic Graph (LangGraph)** — a multi-node decision graph that routes, retrieves, uses tools, evaluates, and responds intelligently
- 🔧 **Specialized Tools** — step-by-step solver, unit converter, concept comparator, study planner, web search, and more
- 🧵 **Conversation Memory** — remembers context across turns within a session
- ✅ **Faithfulness Evaluation** — self-checks answers against source context before showing them

---

## 📸 Features at a Glance

| Feature | Description |
|---|---|
| 💬 Conversational AI | Chat naturally about physics topics |
| 📄 PDF Knowledge Base | Indexes your own physics PDFs via ChromaDB |
| 🔀 Smart Router | Routes each question to the best strategy (retrieve / tool / memory / chat) |
| 🛠️ 7 Built-in Tools | Calculator, Unit Converter, Step Solver, Comparator, Study Planner, Simplifier, Web Search |
| 🧠 Session Memory | Keeps last 6 messages in context for follow-up questions |
| 🔍 Faithfulness Check | Evaluates answer quality; retries if score is below threshold |
| 🌐 Web Search Fallback | Uses DuckDuckGo for out-of-syllabus queries |
| ⚡ Groq LLaMA 3.3 70B | Ultra-fast inference via Groq API |

---

## 🗂️ Topics Covered (Built-in PDFs)

The repo ships with 9 physics PDFs ready to be indexed:

- Simple Harmonic Motion
- Damped Harmonic Motion
- EM Waves Transverse Nature
- Fraunhofer Diffraction
- Inference of Light
- Laser Components
- Maxwell's Equations
- Quantum Process
- Waves and Wave Motion

> You can add your own PDFs to the `pdfs/` folder and they will be automatically indexed on startup.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
[memory_node]          → manages conversation history (last 6 messages)
     │
     ▼
[router_node]          → decides: retrieve / tool / memory_only / chat
     │
     ├──► [retrieval_node]          → queries ChromaDB vector store
     ├──► [skip_retrieval_node]     → skips retrieval for memory/chat routes
     └──► [intent_classifier_node]  → classifies tool intent (calculator/search/etc.)
               │
               ▼
          [tool_node]               → runs the appropriate tool
               │
               ▼
          [answer_node]             → LLM generates final answer
               │
               ▼
          [eval_node]               → faithfulness score check (retries if < 0.7)
               │
               ▼
          [save_node]               → saves assistant reply to memory
               │
               ▼
          Final Answer → User
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11** (the project pins this via `.python-version`)
- A **Groq API Key** (free at [console.groq.com](https://console.groq.com))
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/StudyBuddy.git
cd StudyBuddy
```

---

### 2. Create a Virtual Environment

```bash
# Create and activate a virtual environment
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If you get errors with `chromadb` on Windows, make sure you have the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed.

---

### 4. Set Up Your Groq API Key

Create a `.env` file in the root of the project:

```bash
# On macOS / Linux:
touch .env

# On Windows:
type nul > .env
```

Open the `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get your free API key at [console.groq.com](https://console.groq.com/keys)

---

### 5. Add Your Physics PDFs (Optional)

The `pdfs/` folder already contains 9 pre-loaded physics PDFs. To add your own:

```bash
# Just drop any physics PDF into the pdfs/ folder
cp /path/to/your/notes.pdf pdfs/
```

The agent will automatically index all PDFs in this folder on startup.

---

### 6. Run the App

```bash
streamlit run capstone_streamlit.py
```

The app will open automatically in your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
StudyBuddy/
│
├── agent.py                  # Core agentic logic (LangGraph graph, all nodes, tools)
├── capstone_streamlit.py     # Streamlit frontend / UI
├── requirements.txt          # Python dependencies
├── .python-version           # Python version pin (3.11)
├── .env                      # Your API keys (create this — NOT committed to git)
│
└── pdfs/                     # Physics PDFs that form the knowledge base
    ├── Simple Harmonic Motion.pdf
    ├── Damped Harmonic Motion.pdf
    ├── EM Waves Transverse Nature.pdf
    ├── Fraunhofer Diffraction.pdf
    ├── Inference of Light.pdf
    ├── Laser Components.pdf
    ├── Maxwells Equations.pdf
    ├── Quantum Process.pdf
    └── Waves and Wave Motion.pdf
```

---

## 🛠️ Tools Available to the Agent

When the router decides a **tool** is needed, the intent classifier picks the right one:

| Intent | Tool | What it does |
|---|---|---|
| `calculator` | Calculator | Evaluates numerical physics expressions |
| `convert` | Unit Converter | Converts between physical units with steps |
| `solve` | Step Solver | Provides structured step-by-step derivations |
| `compare` | Comparator | Compares two or more physics concepts side-by-side |
| `plan` | Study Planner | Generates a study roadmap for a topic |
| `simplify` | Simplifier | Explains concepts in plain language with analogies |
| `search` | Web Search | Fetches live info from DuckDuckGo for out-of-syllabus queries |

---

## 💻 Example Questions to Try

```
What is Simple Harmonic Motion?
Explain the difference between interference and diffraction
What are Maxwell's equations?
Solve: A spring of constant 200 N/m stretches by 0.05m. Find the force.
Convert 3 eV to Joules
Create a study plan for wave motion
Explain laser components in simple terms
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repo, branch (`main`), and entry file (`capstone_streamlit.py`)
4. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Click **Deploy** — your app will be live in ~2 minutes!

> 📌 The `pdfs/` folder is included in the repo, so Streamlit Cloud will have access to the knowledge base automatically.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` + `langchain-community` | LLM orchestration and document loading |
| `langchain-groq` | Groq LLaMA 3.3 integration |
| `langgraph` | Agentic state graph execution |
| `chromadb` | Local vector database for RAG |
| `pypdf` | PDF text extraction |
| `duckduckgo-search` | Web search tool |
| `python-dotenv` | `.env` file loader |
| `numpy`, `protobuf` | Supporting libraries |

---

## ⚙️ Configuration

You can tweak these constants at the top of `agent.py`:

```python
FAITHFULNESS_THRESHOLD = 0.7   # Minimum score before retrying answer generation
MAX_EVAL_RETRIES = 2            # Max number of regeneration attempts
PDF_FOLDER = "pdfs"             # Folder to load PDFs from
```

---

## 🔒 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key for LLaMA 3.3 inference |

---

## 📄 License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) for blazing-fast LLaMA inference
- [LangChain](https://langchain.com) & [LangGraph](https://langchain-ai.github.io/langgraph/) for the agentic framework
- [ChromaDB](https://www.trychroma.com/) for the vector store
- [Streamlit](https://streamlit.io) for the UI

---

<p align="center">Built with ❤️ for B.Tech students who are tired of boring textbooks.</p>