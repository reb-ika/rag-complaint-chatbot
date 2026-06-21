🏦 Intelligent Complaint Analysis for Financial Services

A RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights



📌 Project Overview

CrediTrust Financial is a fast-growing digital finance company serving East African markets with over 500,000 users across three countries. They receive thousands of customer complaints every month across credit cards, personal loans, savings accounts, and money transfers.

This project builds an internal AI tool that transforms raw, unstructured complaint data into a strategic asset — enabling Product Managers, Support Teams, and Compliance Officers to ask plain-English questions and receive synthesized, evidence-backed answers in seconds instead of hours.

🎯 Business KPIs

KPIGoal⏱️ Trend IdentificationReduce from days → minutes👥 Non-technical AccessEnable Support & Compliance teams without a data analyst🔍 Proactive InsightsShift from reactive to proactive issue identification


🏗️ System Architecture

Customer Complaints (CFPB Dataset)
         │
         ▼
┌─────────────────────┐
│   Task 1: EDA &     │  → Filter 9.6M rows to 477K relevant complaints
│   Preprocessing     │  → Clean narratives, remove boilerplate
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Task 2: Chunking  │  → Split narratives into 500-char chunks
│   & Embeddings      │  → Generate 384-dim vectors with all-MiniLM-L6-v2
│                     │  → Index 29,024 chunks in FAISS
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Task 3: RAG       │  → Retrieve top-k similar chunks
│   Pipeline          │  → Inject into LLM prompt
│                     │  → Generate synthesized answer
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Task 4: Gradio    │  → Interactive chat interface
│   Chat UI           │  → Source transparency & citation
└─────────────────────┘


📁 Project Structure

rag-complaint-chatbot/
│
├── 📂 .github/
│   └── workflows/
│       └── unittests.yml          # CI/CD pipeline
│
├── 📂 data/
│   ├── raw/                       # Original CFPB dataset
│   └── processed/
│       └── filtered_complaints.csv # Cleaned & filtered dataset (477K rows)
│
├── 📂 vector_store/               # Persisted FAISS index
│   ├── complaints.index           # FAISS vector index (29,024 vectors)
│   ├── chunks_metadata.csv        # Chunk text + metadata
│   └── embeddings.npy             # Raw embedding matrix (29024 × 384)
│
├── 📂 notebooks/
│   ├── eda_preprocessing.ipynb    # Task 1 — EDA and preprocessing
│   ├── chunking_embedding.ipynb   # Task 2 — Chunking and vector store
│   └── eda_plots.png              # EDA visualizations
│
├── 📂 src/
│   └── __init__.py                # RAG pipeline modules (Task 3)
│
├── 📂 tests/
│   └── __init__.py                # Unit tests
│
├── app.py                         # Gradio/Streamlit interface (Task 4)
├── requirements.txt               # Python dependencies
├── README.md                      # You are here!
└── .gitignore


🚀 Getting Started

Prerequisites


Python 3.10+
Git
8GB+ RAM recommended


Installation

1. Clone the repository

bashgit clone https://github.com/reb-ika/rag-complaint-chatbot.git
cd rag-complaint-chatbot

2. Create and activate virtual environment

bashpython -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

3. Install dependencies

bashpip install -r requirements.txt

4. Download the CFPB dataset

Download from the CFPB Official Website and place it in data/raw/complaints.csv


📊 Tasks Progress

✅ Task 1 — EDA and Preprocessing


Loaded and analyzed 9.6 million CFPB complaints
Filtered to 4 target product categories
Removed empty narratives and cleaned text
Saved 477,183 cleaned complaints to filtered_complaints.csv


Dataset Distribution after filtering:

Product CategoryCountPercentageCredit Card188,77939.6%Savings Account154,54432.4%Money Transfer98,44520.6%Personal Loan35,4157.4%Total477,183100%

Key EDA Findings:


📈 Credit Card complaints dominate at 39.6% of all complaints
📝 Average narrative length is 206 words (median: 138 words)
📅 Complaint volume has grown sharply after 2022



✅ Task 2 — Text Chunking, Embedding & Vector Store


Created stratified sample of 10,000 complaints (2,500 per category)
Chunked narratives using RecursiveCharacterTextSplitter

chunk_size = 500 characters
chunk_overlap = 50 characters



Generated embeddings using all-MiniLM-L6-v2 (384 dimensions)
Built and saved FAISS index with 29,024 vectors


Chunking Results:

Product CategoryComplaintsChunksAvg Chunks/ComplaintCredit Card2,5007,6063.0Savings Account2,5007,8283.1Money Transfer2,5006,2972.5Personal Loan2,5007,2932.9Total10,00029,0242.9


🔄 Task 3 — RAG Pipeline (Coming Soon)


Retriever using FAISS similarity search
Prompt engineering with context injection
LLM integration (Mistral / Llama)
Qualitative evaluation on 5-10 test questions



🔄 Task 4 — Gradio Chat Interface (Coming Soon)


Interactive question answering UI
Source chunk display for transparency
Clear/reset functionality





📦 Key Dependencies

pandas==2.x          # Data manipulation
sentence-transformers # Embedding generation
faiss-cpu            # Vector similarity search
langchain-text-splitters # Text chunking
gradio               # Chat UI
scikit-learn         # Stratified sampling

Full list in requirements.txt



