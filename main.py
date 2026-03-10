import os
from typing import TypedDict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader,
    UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
)
from langchain_tavily import TavilySearch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import glob

# ── INITIALIZATION ──────────────────────────────────────────────────────────

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
embeddings = SentenceTransformerEmbeddings()

# ── VECTORSTORE SETUP ───────────────────────────────────────────────────────

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pptx": UnstructuredPowerPointLoader
}

def load_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADERS.get(ext)
    if not loader_cls:
        return []
    try:
        docs = loader_cls(file_path).load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        return docs
    except Exception:
        return []

def build_vectorstore(folder_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
    file_paths = []
    for ext in LOADERS:
        file_paths.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))

    if not file_paths:
        raise FileNotFoundError(f"No supported documents found in '{folder_path}'")

    all_docs = []
    for path in file_paths:
        all_docs.extend(load_document(path))

    if not all_docs:
        raise ValueError("No content could be extracted from the documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

FAISS_INDEX_PATH = "faiss_index"

if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = build_vectorstore("documents")
    vectorstore.save_local(FAISS_INDEX_PATH)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)

# ── AGENT STATE DEFINITION ──────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    answer: str
    route_decision: str
    retrieval_count: int
    chat_history: list[str]
    confidence_score: float
    escalation_flags: list[str]

# ── CONSTANTS ───────────────────────────────────────────────────────────────

MAX_RETRIEVAL_ATTEMPTS = 2
CONFIDENCE_THRESHOLD = 0.6

_indexed_sources = {
    doc.metadata.get("source", "unknown")
    for doc in vectorstore.docstore._dict.values()
}

_available_sources = list(_indexed_sources)

# ── AGENT NODE FUNCTIONS ────────────────────────────────────────────────────

def decide(state: AgentState) -> AgentState:
    sources_list = "\n".join(f"- {s}" for s in _available_sources)
    prompt = f"""You are a routing assistant for a document Q&A system.
The following documents are available to search:
{sources_list}

Classify the user's question into ONE of these categories:
- 'local'  : the question relates to the documents listed above
- 'web'    : the question asks for factual information not covered by the documents
- 'general': the question can be answered from general knowledge

Answer ONLY 'local', 'web', or 'general'.

Question: {state['query']}"""
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    if decision not in ("local", "web", "general"):
        decision = "general"
    state["route_decision"] = decision
    state["retrieval_count"] = 0
    return state

def retrieve(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["query"])
    state["retrieved_docs"] = docs
    state["retrieval_count"] = state.get("retrieval_count", 0) + 1
    return state

def query(state: AgentState) -> AgentState:
    context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])
    prompt = f"""The retrieved documents did not fully answer the question.
Rewrite the question to be more specific.
Return ONLY the rewritten question.

Original question: {state['query']}
Retrieved context: {context}

Rewritten question:"""
    response = llm.invoke(prompt)
    state["query"] = response.content.strip()
    return state

def web_search(state: AgentState) -> AgentState:
    tool = TavilySearch(max_results=4)
    response = tool.invoke(state["query"])
    state["retrieved_docs"] = [
        Document(page_content=r["content"], metadata={"source": r["url"], "title": r.get("title", "")})
        for r in response.get("results", [])
    ]
    return state

def jury_generate(state: AgentState) -> AgentState:
    history_block = ""
    if state.get("chat_history"):
        history_block = "Conversation so far:\n" + "\n".join(state["chat_history"]) + "\n\n"

    query = state["query"]

    if state.get("retrieved_docs"):
        context = "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in state["retrieved_docs"]
        )
        juror1_prompt = f"""{history_block}You are the Citation Juror.
Answer ONLY using explicit statements from the context. Cite sources in brackets.
Context: {context}
Question: {query}
Citation Juror verdict:"""

        juror2_prompt = f"""{history_block}You are the Synthesis Juror.
Synthesize the context to give the most complete answer. Cite sources in brackets.
Context: {context}
Question: {query}
Synthesis Juror verdict:"""

        juror3_prompt = f"""{history_block}You are the Gap Juror.
Identify what the question asks that the context does NOT fully answer. Cite sources for what IS covered.
Context: {context}
Question: {query}
Gap Juror verdict:"""
    else:
        juror1_prompt = f"""{history_block}You are the Citation Juror. Answer from general knowledge.
Question: {query}
Citation Juror verdict:"""
        juror2_prompt = f"""{history_block}You are the Synthesis Juror. Answer from general knowledge comprehensively.
Question: {query}
Synthesis Juror verdict:"""
        juror3_prompt = f"""{history_block}You are the Gap Juror. Answer from general knowledge and note uncertainties.
Question: {query}
Gap Juror verdict:"""

    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(lambda: llm.invoke(juror1_prompt).content.strip())
        f2 = executor.submit(lambda: llm.invoke(juror2_prompt).content.strip())
        f3 = executor.submit(lambda: llm.invoke(juror3_prompt).content.strip())
        verdict1 = f1.result()
        verdict2 = f2.result()
        verdict3 = f3.result()

    judge_prompt = f"""You are the Judge. Synthesise three jurors' verdicts into a single final answer.
Question: {query}
Citation Juror: {verdict1}
Synthesis Juror: {verdict2}
Gap Juror: {verdict3}

Instructions:
1. Produce the best final answer.
2. Preserve source citations.
3. Assess juror disagreement (YES or NO).
4. Assign confidence 0.0–1.0 (1.0=fully supported, 0.0=not supported).

Respond in EXACT format:
JURORS_DISAGREED: YES or NO
CONFIDENCE: 0.0 to 1.0
ANSWER:
[your final answer here]"""

    judge_response = llm.invoke(judge_prompt).content.strip()

    disagreed = False
    confidence = 0.7
    answer_lines = []
    in_answer = False

    for line in judge_response.split("\n"):
        if line.startswith("JURORS_DISAGREED:"):
            disagreed = "YES" in line.upper()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":")[1].strip())
            except ValueError:
                confidence = 0.7
        elif line.startswith("ANSWER:"):
            in_answer = True
        elif in_answer:
            answer_lines.append(line)

    answer = "\n".join(answer_lines).strip() or judge_response

    state["answer"] = answer
    state["confidence_score"] = round(confidence, 2)
    state["escalation_flags"] = ["juror disagreement"] if disagreed else []
    state["chat_history"] = state.get("chat_history", []) + [
        f"User: {query}",
        f"Assistant: {answer}",
    ]
    return state

def escalation_gate(state: AgentState) -> AgentState:
    flags = list(state.get("escalation_flags", []))

    if state["confidence_score"] < CONFIDENCE_THRESHOLD:
        flags.append(f"low confidence ({state['confidence_score']:.0%})")

    if not state.get("retrieved_docs"):
        flags.append("general knowledge only — no sources verified")

    if state.get("retrieved_docs") and "[" not in state["answer"]:
        flags.append("answer contains no citations despite having sources")

    state["escalation_flags"] = flags
    return state

# ── ROUTING ─────────────────────────────────────────────────────────────────

def route_decide(state: AgentState) -> str:
    mapping = {"local": "retrieve", "web": "web_search", "general": "jury_generate"}
    return mapping[state["route_decision"]]

def route_after_retrieve(state: AgentState) -> str:
    if state["retrieval_count"] >= MAX_RETRIEVAL_ATTEMPTS:
        return "web_search"

    if not state.get("retrieved_docs"):
        return "query"

    top_doc = state["retrieved_docs"][0].page_content
    prompt = f"""Is this document relevant to answering the question?
Answer ONLY 'yes' or 'no'.
Question: {state['query']}
Document: {top_doc}"""
    response = llm.invoke(prompt)
    return "jury_generate" if response.content.strip().lower().startswith("yes") else "query"

# ── BUILD AGENT ─────────────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("decide", decide)
    graph.add_node("retrieve", retrieve)
    graph.add_node("query", query)
    graph.add_node("web_search", web_search)
    graph.add_node("jury_generate", jury_generate)
    graph.add_node("escalation_gate", escalation_gate)

    graph.set_entry_point("decide")

    graph.add_conditional_edges("decide", route_decide, {
        "retrieve": "retrieve",
        "web_search": "web_search",
        "jury_generate": "jury_generate",
    })
    graph.add_conditional_edges("retrieve", route_after_retrieve, {
        "jury_generate": "jury_generate",
        "query": "query",
        "web_search": "web_search",
    })
    graph.add_edge("query", "retrieve")
    graph.add_edge("web_search", "jury_generate")
    graph.add_edge("jury_generate", "escalation_gate")
    graph.add_edge("escalation_gate", END)

    return graph.compile()

agent = build_agent()

# ── FASTAPI SETUP ───────────────────────────────────────────────────────────

app = FastAPI(title="Agentic RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source: str
    confidence: float
    flags: list[str]

_SOURCE_LABELS = {
    "local": "Local Documents",
    "web": "Web Search",
    "general": "General Knowledge",
}

_chat_history: list[str] = []

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    global _chat_history

    try:
        result = agent.invoke({
            "query": request.question,
            "retrieved_docs": [],
            "answer": "",
            "route_decision": "general",
            "retrieval_count": 0,
            "chat_history": _chat_history.copy(),
            "confidence_score": 0.0,
            "escalation_flags": [],
        })

        _chat_history = result["chat_history"]

        source = _SOURCE_LABELS.get(result["route_decision"], "Unknown")
        confidence = result["confidence_score"]
        flags = result["escalation_flags"]

        return QueryResponse(
            answer=result["answer"],
            source=source,
            confidence=confidence,
            flags=flags
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/reset-history")
async def reset_history():
    global _chat_history
    _chat_history = []
    return {"message": "Chat history reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
