import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

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

# =====================================================
# Load LLM (Qwen 3B)
# =====================================================

print("Loading Qwen 2.5 3B model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to("cpu")

print("LLM loaded successfully.")

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

class DocumentRequest(BaseModel):
    content: str
    document_id: str = None

# =====================================================
# Smart Chunking Function
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
    doc_id = request.document_id or f"doc_{collection.count()}"

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
# Retrieve Context
# =====================================================

def retrieve_context(query: str, top_k: int = 4):
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

    docs = results["documents"][0]
    if not docs:
        return ""

    context = "\n\n".join([f"- {doc}" for doc in docs])
    return context

# =====================================================
# Chat Endpoint (Qwen Chat Template)
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):

    context = retrieve_context(request.message)

    if context:
        user_prompt = f"""
Use ONLY the provided internal knowledge to answer.

Rules:
- Do not repeat the context.
- Do not invent information.
- Provide a structured numbered list when appropriate.
- Be concise and professional.
- If answer is not found, say:
  "The requested information is not available in the system."

Internal Knowledge:
{context}

Question:
{request.message}
"""
    else:
        user_prompt = request.message

    messages = [
        {
            "role": "system",
            "content": "You are an enterprise HR system assistant."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Proper Qwen chat formatting
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


    return {
        "response": response,
        "context_used": bool(context)
    }
