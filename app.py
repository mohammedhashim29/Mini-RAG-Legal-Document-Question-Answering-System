import os
import streamlit as st
from transformers import pipeline
from utils import *

st.set_page_config(page_title="📄 Legal QA (Mini-RAG)", layout="wide")
st.title("📜 Legal Document Question Answering System (Mini-RAG)")

# Load QA pipeline
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# UI to build index
if st.button("📄 Build Knowledge Base from PDFs"):
    with st.spinner("Reading and indexing PDFs..."):
        docs = extract_text_from_pdfs("data")
        if not docs:
            st.error("No PDF files found in the 'data/' folder.")
        else:
            all_chunks = []
            for doc in docs:
                all_chunks.extend(chunk_text(doc))
            index, _ = build_faiss_index(all_chunks)
            save_index(index, all_chunks)
            st.success(" Knowledge base built and FAISS index saved.")

# Load index if exists
if os.path.exists("faiss_index/index.bin"):
    index, chunks = load_index()

    st.markdown("### 🤔 Ask a legal question from the uploaded leases:")
    question = st.text_input("Type your question here:")

    if question:
        with st.spinner("Retrieving relevant context..."):
            context_chunks = search(question, index, chunks)
            context = " ".join(context_chunks)
            result = qa(question=question, context=context)

        st.markdown(f"###  Answer: `{result['answer']}`")
        with st.expander("📚 Context used"):
            st.write(context)
else:
    st.info("ℹ️ Upload your PDFs in the `data/` folder and click 'Build Knowledge Base'.")
