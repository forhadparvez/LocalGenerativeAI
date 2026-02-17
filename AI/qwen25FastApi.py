import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
import chromadb
import uuid

import requests
from bs4 import BeautifulSoup


# =====================================================
# FastAPI Setup
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5062"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Configuration
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
DEVICE = "cpu"  # change to "cuda" if GPU available

# =====================================================
# Load LLM
# =====================================================

print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(DEVICE)

model.eval()

print("LLM loaded.")

# =====================================================
# Load Embedding Model
# =====================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")

# =====================================================
# Initialize ChromaDB (Persistent)
# =====================================================

print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name="documents"
)

print("RAG system ready.")

# =====================================================
# Request Models
# =====================================================

class ChatRequest(BaseModel):
    message: str
    use_web: bool = False
    web_url: str = None   # NEW
    top_k: int = 4


class DocumentRequest(BaseModel):
    content: str
    document_id: str = None


# =====================================================
# Smart Chunking
# =====================================================

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# =====================================================
# Add Document Endpoint
# =====================================================

@app.post("/add-document")
def add_document(request: DocumentRequest):

    doc_id = request.document_id or f"doc_{uuid.uuid4()}"

    chunks = chunk_text(request.content)

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks
    )

    return {
        "status": "success",
        "chunks_added": len(chunks),
        "document_id": doc_id
    }


# =====================================================
# Retrieve Internal Context
# =====================================================

def retrieve_internal_context(query: str, top_k: int):

    if collection.count() == 0:
        return ""

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return ""

    return "\n\n".join([f"- {doc}" for doc in docs])


# =====================================================
# Web Search Function
# =====================================================

def search_web(query: str, max_results: int = 5):

    web_results = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)

            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")

                web_results.append(
                    f"Title: {title}\n"
                    f"Content: {body}\n"
                    f"Source: {url}"
                )

    except Exception as e:
        print("Web search error:", e)

    return "\n\n".join(web_results)


def fetch_webpage_content(url: str):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "lxml")

        # Remove scripts & styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator="\n")

        # Clean lines
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join([line for line in lines if line])

        # Limit very large pages
        return cleaned[:8000]

    except Exception as e:
        print("Webpage fetch error:", e)
        return ""


# =====================================================
# Build Prompt
# =====================================================

def build_prompt(question: str, context: str):

    return f"""
You are an enterprise HR system assistant.

Use ONLY the provided knowledge to answer.

Rules:
- Do not repeat the context.
- Do not invent information.
- Provide a structured numbered list when appropriate.
- Be concise and professional.
- If answer is not found, say:
  "The requested information is not available in the system."

Knowledge:
{context}

Question:
{question}
"""


# =====================================================
# Chat Endpoint
# =====================================================


    internal_context = retrieve_internal_context(
        request.message,
        request.top_k
    )

    web_context = ""

    if request.use_web:
        web_context = search_web(request.message)

    # Combine context safely
    combined_context = ""

    if internal_context:
        combined_context += "Internal Knowledge:\n" + internal_context + "\n\n"

    if web_context:
        combined_context += "Web Knowledge:\n" + web_context

    if not combined_context:
        combined_context = ""

    prompt = build_prompt(request.message, combined_context)

    messages = [
        {"role": "system", "content": "You are an enterprise HR system assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return {
        "response": response,
        "internal_context_used": bool(internal_context),
        "web_used": bool(web_context)
    }

@app.post("/chat")
def chat(request: ChatRequest):

    internal_context = retrieve_internal_context(
        request.message,
        request.top_k
    )

    webpage_context = ""

    # Fetch specific webpage if URL provided
    if request.web_url:
        webpage_text = fetch_webpage_content(request.web_url)

        if webpage_text:
            # Optional: chunk & limit
            chunks = chunk_text(webpage_text, chunk_size=1000, overlap=200)
            webpage_context = "\n\n".join(chunks[:3])  # limit to first 3 chunks

    combined_context = ""

    if internal_context:
        combined_context += "Internal Knowledge:\n" + internal_context + "\n\n"

    if webpage_context:
        combined_context += "Webpage Content:\n" + webpage_context

    prompt = build_prompt(request.message, combined_context)

    messages = [
        {"role": "system", "content": "You are an enterprise HR system assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return {
        "response": response,
        "internal_context_used": bool(internal_context),
        "webpage_used": bool(webpage_context)
    }

# =====================================================
# Health Check
# =====================================================

@app.get("/")
def root():
    return {"status": "Hybrid RAG + Web Search system running"}
