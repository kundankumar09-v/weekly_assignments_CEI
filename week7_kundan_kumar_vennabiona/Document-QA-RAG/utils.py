"""
utils.py  -  Utility functions for Document QA RAG System
"""

import re

def clean_extracted_text(text):
    """
    Process the extracted text by removing unnecessary whitespace
    and cleaning invalid characters.
    """
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text

def get_total_chars(docs):
    """Calculate the total number of characters across all documents."""
    return sum(len(doc.page_content) for doc in docs)

def print_ingestion_logs(num_pages, total_chars, num_chunks, embed_model, embed_dim, vector_db):
    """Print validation logs to the terminal for the ingestion phase."""
    print("\n-----------------------------------")
    print("Document loaded successfully")
    print(f"Number of pages: {num_pages}")
    print(f"Number of extracted characters: {total_chars}")
    print(f"Number of chunks created: {num_chunks}")
    print(f"Embedding model: {embed_model}")
    print(f"Embedding dimensions: {embed_dim}")
    print(f"Vector database initialized: {vector_db}")
    print("-----------------------------------\n")

def print_generation_logs(query, retrieved_count, elapsed):
    """Print validation logs to the terminal for the generation phase."""
    print("\n-----------------------------------")
    print(f"User question: {query}")
    print(f"Number of retrieved chunks: {retrieved_count}")
    print("Response generated successfully")
    print(f"Execution time: {elapsed} seconds")
    print("-----------------------------------\n")
