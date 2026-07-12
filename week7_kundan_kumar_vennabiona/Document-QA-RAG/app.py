"""
app.py - Streamlit application for Document QA RAG System
"""

import streamlit as st
import rag
import utils

# ──────────────────────────────────────────────────────
# User Interface Config
# ──────────────────────────────────────────────────────
st.set_page_config(page_title="Document QA RAG System", layout="wide")

st.title("Document Question Answering System")
st.write("Upload your documents below and ask questions. Answers are generated only from the uploaded context.")

# ──────────────────────────────────────────────────────
# Document Ingestion Section
# ──────────────────────────────────────────────────────
st.subheader("1. Upload Documents")
col1, col2 = st.columns(2)
with col1:
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
with col2:
    txt_files = st.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True)

# Optional dataset loader
use_hf = st.checkbox("Optional: Load a Hugging Face dataset")
hf_dataset = ""
if use_hf:
    hf_dataset = st.text_input("Hugging Face Dataset Name (e.g., squad)", "")

# ──────────────────────────────────────────────────────
# Processing Documents
# ──────────────────────────────────────────────────────
if st.button("Process Documents"):
    docs = []
    
    # Load files
    for f in pdf_files:
        with st.spinner(f"Loading {f.name}..."):
            docs.extend(rag.load_pdf(f))
            
    for f in txt_files:
        with st.spinner(f"Loading {f.name}..."):
            docs.extend(rag.load_txt(f))

    if use_hf and hf_dataset:
        with st.spinner("Loading dataset..."):
            docs.extend(rag.load_hf_dataset(hf_dataset, "train", "text"))

    if not docs:
        st.error("Please upload at least one document.")
        st.stop()

    total_chars = utils.get_total_chars(docs)

    # Process Text
    with st.spinner("Processing text and creating chunks..."):
        chunks = rag.split_text(docs)

    # Generate Embeddings & Vector Store
    with st.spinner("Initializing embedding model and vector database..."):
        embeddings = rag.create_embeddings()
        store = rag.create_vector_store(chunks, embeddings)
        tokenizer, model = rag.load_lm()

    # Save to session state
    st.session_state["store"] = store
    st.session_state["chunks"] = chunks
    st.session_state["tokenizer"] = tokenizer
    st.session_state["model"] = model
    
    metrics = {
        "num_docs": len(pdf_files) + len(txt_files) + (1 if use_hf and hf_dataset else 0),
        "total_chars": total_chars,
        "num_chunks": len(chunks),
        "embed_model": rag.EMBED_MODEL_NAME,
        "embed_dim": rag.EMBED_DIM,
        "vector_db": "FAISS",
        "lm": rag.LM_MODEL_NAME,
        "top_k": rag.TOP_K
    }
    st.session_state["metrics"] = metrics

    # Terminal Validation Logs
    utils.print_ingestion_logs(
        num_pages=len(docs),
        total_chars=total_chars,
        num_chunks=len(chunks),
        embed_model=rag.EMBED_MODEL_NAME,
        embed_dim=rag.EMBED_DIM,
        vector_db="FAISS"
    )

    st.success("Documents processed successfully! You can now ask questions.")

st.markdown("---")

# ──────────────────────────────────────────────────────
# Query & Answer Section
# ──────────────────────────────────────────────────────
if "store" in st.session_state:
    st.subheader("2. Ask a Question")
    query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            store = st.session_state["store"]
            tokenizer = st.session_state["tokenizer"]
            model = st.session_state["model"]

            with st.spinner("Retrieving relevant information..."):
                result = rag.retrieve_chunks(store, query)

            with st.spinner("Generating answer..."):
                answer, elapsed = rag.generate_answer(result, query, tokenizer, model)

            # Display Answer
            st.subheader("Generated Answer")
            st.write(answer)

            # Display Retrieved Context
            st.subheader("Retrieved Context")
            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(result):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(chunk.page_content)
                    st.markdown("---")

            # Terminal Validation Logs
            utils.print_generation_logs(
                query=query,
                retrieved_count=len(result),
                elapsed=elapsed
            )

st.markdown("---")

# ──────────────────────────────────────────────────────
# System Metrics Report
# ──────────────────────────────────────────────────────
if "metrics" in st.session_state:
    st.subheader("System Metrics Report")
    m = st.session_state["metrics"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Number of uploaded documents:** {m['num_docs']}")
        st.markdown(f"- **Total extracted characters:** {m['total_chars']}")
        st.markdown(f"- **Number of chunks:** {m['num_chunks']}")
        st.markdown(f"- **Retrieval Top-K value:** {m['top_k']}")
    with col2:
        st.markdown(f"- **Embedding model:** {m['embed_model']}")
        st.markdown(f"- **Embedding dimensions:** {m['embed_dim']}")
        st.markdown(f"- **Vector database:** {m['vector_db']}")
        st.markdown(f"- **Language model:** {m['lm']}")
