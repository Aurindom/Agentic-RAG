# Agentic RAG System

End-to-end Retrieval-Augmented Generation pipeline built with LangGraph and Claude. The agent autonomously decides whether to search local documents, query the web, or answer from general knowledge — without any manual routing.

---

## What it does

Every question is classified into one of three paths:

- **Local** — searches an indexed FAISS vector store built from your own documents
- **Web** — calls Tavily to retrieve live search results when the answer is not in local docs
- **General** — answers directly from the LLM when no retrieval is needed (math, definitions, casual questions)

If local retrieval returns irrelevant chunks, the agent rewrites the query and retries. After exhausting local attempts it falls back to web search automatically.

All answers include per-source citations — filenames for local docs, URLs for web results.

---

## Features

- Three-way autonomous routing (local docs / web search / general knowledge)
- Self-correcting retrieval loop with query reformulation
- Tavily web search fallback for questions outside the local document set
- MMR retrieval for diverse, non-repetitive chunks
- Persistent FAISS index — built once, reloaded on every run
- Multi-turn conversation memory with cited answers across sessions
- Supports PDF, DOCX, PPTX, TXT, CSV, MD documents

---

## Stack

| Layer | Tool |
|---|---|
| Agent framework | LangGraph |
| LLM | Claude Sonnet (Anthropic) |
| Embeddings | `all-mpnet-base-v2` (HuggingFace, runs locally) |
| Vector store | FAISS |
| Web search | Tavily |
| Document loaders | LangChain Community |

---

## Setup

**1. Clone and install dependencies**
```bash
pip install langgraph langchain langchain-anthropic langchain-community \
            faiss-cpu sentence-transformers tavily-python python-dotenv \
            pypdf docx2txt unstructured
```

**2. Add your API keys to a `.env` file**
```
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

**3. Drop your documents into the `documents/` folder**

Supported: `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`, `.md`

**4. Run all cells in `agenticrag.ipynb` top to bottom**

The FAISS index builds automatically on first run and is reused on subsequent runs.

---

## Usage

```python
# Single question — routes to local docs, web, or general knowledge automatically
answer = run_agent("What is the EU inflation rate in 2024?")
print(answer)

# Multi-turn conversation
_chat_history.clear()
print(run_agent("What does the document say about logistic regression?"))
print(run_agent("How does that compare to linear regression?"))
```

Watch the logs to see which path each question takes:
```
[decide] route=local
[retrieve] got 4 chunks (attempt 1)
[route_after_retrieve] → generate
[generate] done.
```

---

## Agent Flow

```
decide (classifies question)
    |
    |-- "local"   --> retrieve --> relevant? --> generate
    |                    |                          |
    |                    | no, retry               END
    |                    |
    |                 query (reformulate) --> retrieve --> max attempts --> web_search
    |
    |-- "web"     --> web_search --> generate --> END
    |
    |-- "general" --> generate --> END
```

---

## Adding new documents

1. Drop files into `documents/`
2. Delete the `faiss_index/` folder
3. Restart the kernel and run all cells
