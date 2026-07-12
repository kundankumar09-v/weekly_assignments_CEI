# Document Question Answering System (RAG)

## Project Overview
This is a complete, production-ready, beginner-friendly Retrieval-Augmented Generation (RAG) based Document Question Answering System. The application answers user questions only from the uploaded documents (PDF or TXT) by retrieving relevant information and generating a grounded response.

## Objectives
- Build a robust RAG pipeline using open-source tools.
- Extract and process text from custom documents.
- Generate context-aware answers without hallucinating or using outside knowledge.

## Key Concepts
### Retrieval
The process of finding relevant chunks of information from a knowledge base (in this case, uploaded documents stored in FAISS) based on a user's query.

### Augmentation
Combining the retrieved chunks with the original user query to create a comprehensive prompt that provides the LLM with the necessary context.

### Generation
Using a Language Model (`google/flan-t5-base`) to read the augmented prompt and output a clear, concise, and logically ordered answer.

## System Architecture
1. **Document Ingestion:** PDFs and TXT files are uploaded and text is extracted.
2. **Text Processing:** Text is cleaned and split into 500-character chunks.
3. **Embedding:** Chunks are converted into vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Database:** Embeddings are stored in FAISS for fast similarity search.
5. **Retrieval:** The user's query is embedded and matched against FAISS to find the top 3 most relevant chunks.
6. **Generation:** `gemini-pro` generates a response strictly based on the retrieved context.

## Workflow
User Uploads File -> Text Extraction -> Chunking -> Embedding -> Vector DB -> User Asks Question -> Query Embedded -> Context Retrieved -> Prompt Augmented -> Answer Generated.

## Components Used
- Document Ingestion (PyPDFLoader)
- Text Splitter (RecursiveCharacterTextSplitter)
- Embeddings (HuggingFaceEmbeddings)
- Vector Store (FAISS)
- LLM (AutoModelForSeq2SeqLM)

## Technologies Used
- Streamlit
- LangChain
- FAISS
- Hugging Face Transformers
- Sentence Transformers
- PyPDF
- Torch

## Folder Structure
```
Document-QA-RAG/
│
├── app.py
├── rag.py
├── utils.py
├── requirements.txt
├── README.md
├── sample_data/
└── screenshots/
```

## Installation
1. Clone the repository.
2. Ensure you have Python installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the application:
   ```bash
   python -m streamlit run app.py
   ```
2. Upload your PDF or TXT files.
3. Wait for the success message.
4. Enter your question and click "Ask".

## Example Questions
- "What is the main topic of the document?"
- "What are the key conclusions?"
- "Summarize the findings in bullet points."

## Key Learnings
- Understanding the relationship between chunk sizes and token limits.
- Implementing FAISS for efficient similarity search.
- Controlling an LLM to prevent hallucination by strictly grounding its prompt.
- Creating modular, beginner-friendly Python applications using Streamlit.

## Conclusion
This project successfully demonstrates how to construct a complete RAG system using entirely free and open-source models, satisfying all requirements of the academic assignment.
