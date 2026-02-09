# Pathfinder RAG

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that produces
broker-style mortgage suggestions based on a client’s **age range** and **income type**.
All recommendations are strictly grounded in a single internal training document
(`Pathfinder AI - Training Materials.docx`).


---

## Design Principles
- **Heading-based document chunking** to preserve semantic meaning
- **Hybrid retrieval** using BM25 keyword matching and embedding-based cosine similarity
- **Grounded generation** using a local LLM (Ollama)
- Pipeline-level validation and guardrails
- Full traceability via retrieved chunk citations

---

## Folder Structure
``` text
data/
Pathfinder AI - Training Materials.docx
src/
config.py
ingest.py
retriever.py
generator.py
pipeline.py
demo.py
requirements.txt
```

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
Start Ollama and ensure the model is available:

ollama pull llama3:8b
Run the demo:

python demo.py
Test Cases
The following test cases are executed to validate retrieval and grounded generation:

18–24 + Salary

25–34 + Hourly + Overtime

45–54 + Self-Employed

Each run returns structured JSON including:

broker suggestions

follow-up questions

retrieved evidence with chunk IDs

