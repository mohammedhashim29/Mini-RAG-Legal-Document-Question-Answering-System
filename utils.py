import os
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdfs(pdf_dir="data"):
    documents = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_dir, file))
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            documents.append(full_text)
    return documents

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def save_index(index, chunks, path="faiss_index"):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.bin"))
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

def load_index(path="faiss_index"):
    index = faiss.read_index(os.path.join(path, "index.bin"))
    with open(os.path.join(path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query, index, chunks, top_k=10):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in I[0]]
