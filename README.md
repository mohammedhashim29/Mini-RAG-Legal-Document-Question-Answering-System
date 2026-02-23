# Mini-RAG-Legal-Document-Question-Answering-System
A lightweight Retrieval-Augmented Generation (Mini-RAG) system for answering legal questions from PDF documents using FAISS, MiniLM embeddings, and DistilBERT — fully CPU-based.

This project implements a lightweight Retrieval-Augmented Generation (Mini-RAG) pipeline for answering legal questions from PDF lease documents.

The system extracts text from preloaded PDFs, builds a semantic search index using MiniLM embeddings + FAISS, and uses a DistilBERT Question Answering model to extract accurate answers from retrieved document context — all running locally on CPU.

🚀 Features
Reads legal PDFs from /data folder
Splits documents into semantic chunks
Generates embeddings using all-MiniLM-L6-v2
Uses FAISS for fast semantic similarity search
Uses DistilBERT (SQuAD) for extractive QA
Interactive UI built with Streamlit
Runs fully on CPU (16GB RAM, no GPU required)
Displays supporting context for transparency
Architecture (Mini-RAG Pipeline)

Extract text from PDFs
Chunk text into smaller passages
Convert chunks into embeddings (MiniLM)
Store embeddings in FAISS index
Embed user query
Retrieve Top-K relevant chunks
Pass question + context to DistilBERT QA model
Display answer + context

🛠 Tech Stack
Python
Streamlit
Sentence Transformers (MiniLM)
FAISS
Hugging Face Transformers (DistilBERT QA)
PyPDF2


<img width="940" height="640" alt="image" src="https://github.com/user-attachments/assets/933335f1-2e0c-4783-9ed8-e8e23479f78c" />
