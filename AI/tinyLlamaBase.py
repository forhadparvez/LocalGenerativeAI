import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
embedding_model_name = "all-MiniLM-L6-v2"
chroma_db_path = "./chroma_db"

print("Loading language model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading embedding model...")
embedding_model = SentenceTransformer(embedding_model_name)

print("Initializing ChromaDB...")
# Use PersistentClient for persistent storage
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def add_document_to_rag(content: str, document_id: str = None):
    """Add a document to the RAG knowledge base"""
    doc_id = document_id or f"doc_{collection.count()}"
    
    # Split document into chunks (simple sentence splitting)
    chunks = [chunk.strip() for chunk in content.split(".") if chunk.strip()]
    
    embeddings = embedding_model.encode(chunks)
    
    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    
    print(f"✓ Added {len(chunks)} chunks from document: {doc_id}")

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant documents from the knowledge base"""
    if collection.count() == 0:
        return ""
    
    query_embedding = embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    context = "\n".join(results["documents"][0]) if results["documents"][0] else ""
    return context

print("Model loaded. Start chatting!\n")

# Example: Add sample documents to RAG before starting
print("Adding sample documents to knowledge base...")
sample_docs = {
    "python_basics": "Python is a programming language. Python supports object-oriented programming. Python has many libraries.",
    "ai_concepts": "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks. RAG combines retrieval and generation."
}

for doc_id, content in sample_docs.items():
    add_document_to_rag(content, doc_id)

print("\nStart chatting (type 'exit' or 'quit' to stop):\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    print("Bot:", response, "\n")
