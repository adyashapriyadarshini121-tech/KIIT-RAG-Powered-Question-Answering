
# KIIT RAG-Powered Question Answering

## Project Overview

KIIT RAG-Powered Question Answering is a Retrieval-Augmented Generation system that answers questions using information retrieved from KIIT academic documents.

## Features

- Question answering from KIIT documents
- Semantic document retrieval
- ChromaDB vector database
- Sentence Transformer embeddings
- Groq LLM for answer generation
- Source document display

## Technology Stack

- Python
- Streamlit
- Sentence Transformers
- ChromaDB
- Groq API
- PyPDF

## Knowledge Base

The system uses KIIT academic documents including:

- Academic Regulations
- Student Handbook
- KIIT Academic Calendar
- School of Computer Engineering syllabus

## How It Works

PDF Documents
→ Text Extraction
→ Chunking
→ Embeddings
→ ChromaDB
→ Question Retrieval
→ Groq LLM
→ Answer + Sources

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
