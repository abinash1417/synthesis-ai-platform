# 🔬 Synthesis — AI Research & Knowledge Platform

Synthesis is a multi-agent RAG (Retrieval-Augmented Generation) system that lets you upload documents, then get AI-researched, cited answers to your questions — powered by a team of specialized AI agents that search, write, and review before responding.

## What it does

Upload one or more PDFs, ask a question, and Synthesis:

1. **Researches** — an AI agent searches your documents for relevant information (or reads a full document when asked to review/summarize it)
2. **Writes** — a second agent turns the research into a clear, well-structured answer
3. **Reviews** — a third agent checks the answer for accuracy and clarity, sending it back for revision if it doesn't meet the bar (up to 2 revision cycles)

The result is an answer that's genuinely grounded in your uploaded documents, with source citations — not just the model's general knowledge.

## Why this is different from basic RAG

Most simple RAG demos just retrieve text chunks and paste them into a prompt. Synthesis instead gives an **agent** the ability to decide, on its own, which of two tools to use:

- `search_documents` — for specific, targeted questions ("What does the report say about Q3 revenue?")
- `get_full_document` — for whole-document review questions ("How strong is this resume?")

This distinction was discovered and fixed during development — an early version of the agent tried to answer whole-document questions using fragment search, and failed to recognize the document was even loaded. Solving this required rethinking the tool design, not just tweaking a prompt.

## Architecture

```
User uploads PDFs
       |
       v
Documents chunked -> embedded -> stored in ChromaDB
       |
       v
User asks a question
       |
       v
Researcher Agent (decides: search fragments, or read whole document?)
       |
       v
Writer Agent (drafts a clear answer from the research)
       |
       v
Reviewer Agent (checks quality)
       |
       +--- fails ---> back to Writer (max 2 revisions)
       |
       v passes
Final answer + sources shown to user
```

## Tech Stack

- **LLM:** Groq (`openai/gpt-oss-120b`) via LangChain
- **Orchestration:** LangGraph (StateGraph, multi-agent workflow with conditional routing)
- **Vector Store:** ChromaDB (persistent, local)
- **Frontend:** Streamlit
- **Resilience:** `tenacity` for automatic retry on API rate limits

## Project Structure

```
synthesis/
├── app.py                  # Streamlit UI
├── config.py                # Centralized settings
├── document_processor.py    # KnowledgeBase class - chunking, storage, retrieval
├── tools.py                  # Agent-callable tools
├── agents.py                 # ResearchPipeline class - multi-agent graph
├── static/
│   └── style.css              # UI styling
└── requirements.txt
```

## Running Locally

```bash
git clone https://github.com/abinash1417/synthesis-ai-platform.git
cd synthesis-ai-platform
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Create a `.env` file with:

```
GROQ_API_KEY=your_key_here
```

Then run:

```bash
streamlit run app.py
```

## Known Limitations

- Groq's free tier has an 8,000 token/minute limit — very large documents or rapid successive questions may trigger automatic retries (handled gracefully, but can slow response time)
- Currently supports PDF uploads only

## What I'd Add Next

- Support for additional file types (DOCX, TXT)
- Hybrid search (keyword + semantic) for improved retrieval on exact-match queries
- LangSmith tracing for observability into agent decision-making