# RAG Complaint Chatbot

A Retrieval-Augmented Generation (RAG) demo for financial services complaint analysis.

This project processes CFPB complaint data, performs EDA and preprocessing, chunks complaint narratives, generates embeddings, and builds a FAISS index for fast retrieval.

---

## 🚀 Project Overview

- Clean and filter raw complaint data
- Chunk long complaint narratives into searchable passages
- Generate semantic embeddings using `sentence-transformers`
- Index data with `faiss` for similarity search
- Build a RAG-ready pipeline for question answering

---

## 📦 Repository Structure

```text
rag-complaint-chatbot/
├── .github/                   # CI/CD workflows
├── data/                      # Dataset storage
│   ├── raw/                   # Original source data
│   └── processed/             # Cleaned and filtered data
├── notebooks/                 # Analysis and experimentation notebooks
├── src/                       # Core application modules
├── tests/                     # Unit tests and validation
├── app.py                     # Chat interface entrypoint
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files excluded from version control
```

---

## 🧭 What’s Included

- `notebooks/eda_preprocessing.ipynb` — EDA, filtering, and text cleaning
- `notebooks/chunking_embedding.ipynb` — Chunking, embedding generation, and FAISS indexing
- `app.py` — Interface launcher for the chat application
- `requirements.txt` — Required Python packages

---

## 🧰 Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

1. Download the CFPB dataset and place it in `data/raw/complaints.csv`
2. Run the preprocessing notebook to generate the cleaned dataset
3. Run the chunking and embedding notebook to build the vector index
4. Start the app with `python app.py`

---

## 📌 Notes

- Do not commit large raw data files or generated vector store files to GitHub.
- Keep `data/raw/` and `vector_store/` local and untracked.
- Use `README.md` as the primary project documentation for collaborators.

---

## ✅ Current Status

- Task 1: EDA and preprocessing completed
- Task 2: Chunking, embedding, and FAISS indexing completed
- Task 3: RAG pipeline planned
- Task 4: Gradio chat interface planned
