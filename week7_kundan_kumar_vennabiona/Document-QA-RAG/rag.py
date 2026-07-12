"""
rag.py - RAG pipeline for Document Question Answering System
"""

import os
import time
import tempfile
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_groq import ChatGroq

def get_api_key():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None

API_KEY = get_api_key()

import utils

# ──────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM        = 384
LM_MODEL_NAME    = "llama-3.1-8b-instant"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 100
TOP_K            = 6

# ──────────────────────────────────────────────────────
# 1. Document Ingestion
# ──────────────────────────────────────────────────────
def load_pdf(uploaded_file):
    """Load text from a PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return docs

def load_txt(uploaded_file):
    """Load text from a TXT file."""
    text = uploaded_file.read().decode("utf-8")
    docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
    return docs

def load_hf_dataset(dataset_name, split_name, text_column):
    """(Optional) Load text from a Hugging Face dataset."""
    import datasets
    data = datasets.load_dataset(dataset_name, split=split_name)
    docs = []
    for row in data:
        text = str(row.get(text_column, ""))
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": dataset_name}))
        if len(docs) >= 500:
            break
    return docs

# ──────────────────────────────────────────────────────
# 2. Text Processing
# ──────────────────────────────────────────────────────
def split_text(docs):
    """Clean and split text into manageable chunks."""
    for doc in docs:
        doc.page_content = utils.clean_extracted_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return chunks

# ──────────────────────────────────────────────────────
# 3. Embedding Generation
# ──────────────────────────────────────────────────────
def create_embeddings():
    """Load the sentence-transformers model for embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return embeddings

# ──────────────────────────────────────────────────────
# 4. Vector Database
# ──────────────────────────────────────────────────────
def create_vector_store(chunks, embeddings):
    """Store all chunk embeddings in FAISS."""
    store = FAISS.from_documents(chunks, embeddings)
    return store

# ──────────────────────────────────────────────────────
# 5 & 6. Query Processing and Retrieval Module
# ──────────────────────────────────────────────────────
def retrieve_chunks(store, query):
    """Retrieve the most relevant chunks from FAISS based on the query."""
    result = store.similarity_search(query, k=TOP_K)
    return result

# ──────────────────────────────────────────────────────
# 7. Answer Generation
# ──────────────────────────────────────────────────────
def load_lm():
    """Load the Groq LLM for fast and effective generation."""
    if not API_KEY or API_KEY == "your_groq_api_key_here":
        print("WARNING: Valid GROQ_API_KEY not found in parent directory .env file.")
    llm = ChatGroq(
        model_name=LM_MODEL_NAME,
        api_key=API_KEY,
        temperature=0.3
    )
    return None, llm

def generate_answer(chunks, query, tokenizer, model):
    """Generate answers only from the retrieved context using Groq."""
    context = "\n\n".join([c.page_content for c in chunks])

    prompt = (
        "You are an expert analytical assistant. Your task is to provide a highly detailed, "
        "comprehensive, and well-structured answer to the user's question.\n"
        "Explain concepts in a logical order. Start with the direct answer. "
        "Then provide supporting explanation. Use bullet points whenever appropriate. "
        "Highlight important terms using Markdown bold formatting.\n"
        "CRITICAL RULE: You must base your answer *strictly and exclusively* on the provided context below. "
        "Do not use outside knowledge. If the context does not contain the answer, say exactly: "
        "'I could not find the answer in the uploaded documents.'\n\n"
        f"Context Documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Detailed Answer:"
    )

    start = time.time()
    try:
        response = model.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    elapsed = round(time.time() - start, 2)

    if not answer or len(answer) < 5:
        answer = "I could not find the answer in the uploaded documents."

    return answer, elapsed
