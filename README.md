# Agentic RAG System

End-to-end Retrieval-Augmented Generation pipeline built with LangGraph and Claude. The agent autonomously decides whether to search local documents, query the web, or answer from general knowledge — and validates every answer through a multi-agent jury before returning it.

---

## What it does

Every question is classified into one of three paths:

- **Local** — searches an indexed FAISS vector store built from your own documents
- **Web** — calls Tavily to retrieve live search results when the answer is not in local docs
- **General** — answers directly from the LLM when no retrieval is needed (math, definitions, casual questions)

If local retrieval returns irrelevant chunks, the agent rewrites the query and retries. After exhausting local attempts it falls back to web search automatically.

Every answer then passes through a **3-juror deliberation + judge synthesis**, and an **escalation gate** that flags low-confidence or unsourced answers before they reach you.

---

## Features

- Three-way autonomous routing (local docs / web search / general knowledge)
- Self-correcting retrieval loop with query reformulation
- Tavily web search fallback for questions outside the local document set
- MMR retrieval for diverse, non-repetitive chunks
- Persistent FAISS index — built once, reloaded on every run
- Multi-agent jury: 3 specialist jurors run in **parallel** + judge LLM produce a confidence-scored final answer
- Escalation gate: flags low-confidence, uncited, or disputed answers
- Every response shows source type and confidence level
- Multi-turn conversation memory with per-source citation across sessions
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
| Web API | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes (Minikube locally, EKS/GKE/AKS in cloud) |

---

## Jury System

After retrieval, instead of a single LLM call, three specialist jurors analyse the question **concurrently** using `ThreadPoolExecutor`:

| Juror | Role |
|---|---|
| Citation Juror | Answers only from explicit source statements. Flags anything not directly in the context. |
| Synthesis Juror | Connects the dots across retrieved chunks for the most complete answer. |
| Gap Juror | Identifies what the context fails to answer. Surfaces missing information explicitly. |

All three jurors fire simultaneously — only the **Judge LLM** waits for their results before synthesising a final answer and confidence score (0.0–1.0). This cuts jury latency roughly in half compared to sequential execution.

The **Escalation Gate** checks for:
- Confidence below 60%
- No sources retrieved (general knowledge only)
- Juror disagreement
- Answer contains no citations despite having sources

Flagged answers are returned with a visible warning so you know not to rely on them blindly.

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
answer = run_agent("What is the difference between logistic and linear regression?")
print(answer)
```

Every response prints a metadata block before the answer:

```
============================================================
  Source     : Local Documents
  Confidence : 87%
============================================================
```

Or if flagged:

```
============================================================
  Source     : Web Search
  Confidence : 45%
  FLAGGED    : low confidence (45%), juror disagreement
============================================================
```

---

## Agent Flow

```
decide (classifies question)
    |
    |-- "local"   --> retrieve --> relevant? --> jury_generate
    |                    |                            |
    |                    | no, retry            escalation_gate
    |                    |                            |
    |                 query (reformulate)            END
    |                    |
    |                 retrieve --> max attempts --> web_search
    |
    |-- "web"     --> web_search --> jury_generate --> escalation_gate --> END
    |
    |-- "general" --> jury_generate --> escalation_gate --> END
```

---

## Kubernetes Deployment (Production)

### Local Deployment (Minikube)

**Prerequisites:** Docker, Minikube, kubectl installed

**1. Start Minikube cluster**
```bash
minikube start
kubectl get nodes
```

**2. Build Docker image**
```bash
docker build -t agentic-rag:latest .
minikube image load agentic-rag:latest
```

**3. Create Kubernetes secret for API keys**
```powershell
$env_content = Get-Content ".env" -Raw
$anthropic_key = ($env_content | Select-String "ANTHROPIC_API_KEY=(.+)").Matches.Groups[1].Value
$tavily_key = ($env_content | Select-String "TAVILY_API_KEY=(.+)").Matches.Groups[1].Value

kubectl create secret generic api-keys `
  --from-literal=anthropic-key=$anthropic_key `
  --from-literal=tavily-key=$tavily_key
```

**4. Deploy to Kubernetes**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

**5. Access the service**
```bash
minikube service agentic-rag --url
curl http://127.0.0.1:PORT/query -X POST -H "Content-Type: application/json" -d '{"question":"Your question here"}'
```

### Cloud Deployment

The same deployment manifests work on AWS EKS, Google GKE, or Azure AKS — only the image storage changes:

1. Push image to cloud container registry (e.g., ECR, GCR, ACR)
2. Update `deployment.yaml` with registry URL
3. Apply same `kubectl apply` commands to your cloud cluster

### Files

- **`Dockerfile`** — Builds Docker image with Python 3.11, FastAPI, and all dependencies
- **`main.py`** — FastAPI wrapper; loads agent and serves `/query` endpoint
- **`deployment.yaml`** — Kubernetes deployment manifest; specifies replicas, image, and port
- **`service.yaml`** — Kubernetes service manifest; exposes app via NodePort on port 30000

---

## Adding new documents

1. Drop files into `documents/`
2. Delete the `faiss_index/` folder
3. Restart the kernel and run all cells
