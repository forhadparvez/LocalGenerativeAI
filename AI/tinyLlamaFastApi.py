import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5062"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"

# =========================
# Load LLM
# =========================
print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to("cpu")

# =========================
# Load Embedding Model
# =========================
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# =========================
# Initialize ChromaDB
# =========================
print("Initializing ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name="documents"
)

print("RAG system ready!")

# =========================
# Models
# =========================
class ChatRequest(BaseModel):
    message: str

class DocumentRequest(BaseModel):
    content: str
    document_id: str = None


# =========================
# Better Chunking
# =========================
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# =========================
# Add Document
# =========================
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


# =========================
# Retrieve Context
# =========================
def retrieve_context(query: str, top_k: int = 3):
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


# =========================
# Chat Endpoint (Improved Prompt)
# =========================
@app.post("/chat")
def chat(request: ChatRequest):

    context = retrieve_context(request.message)

    if context:
        prompt = f"""
        You are an enterprise HR system assistant.

        Answer the user's question using ONLY the internal knowledge provided.

        Rules:
        - Do NOT repeat the internal knowledge.
        - Do NOT mention "Internal Knowledge".
        - Provide a clean, structured answer.
        - Use bullet points when appropriate.
        - Be concise and professional.
        - If information is not found, say:
        "The requested information is not available in the system."

        Internal Knowledge:
        {context}

        User Question:
        {request.message}

        Final Answer:
        """


    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()


    return {
        "response": response,
        "context_used": bool(context)
    }
