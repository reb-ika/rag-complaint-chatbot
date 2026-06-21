"""
app.py
CrediTrust Financial - Complaint Analyzer
Deep teal & gold, East African inspired design
"""

import sys
sys.path.append(r'C:\Users\pc\rag-complaint-chatbot')

import os
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

from src.retriever import load_vector_store, retrieve

# ── Load environment variables ──────────────────────────────
load_dotenv(r'C:\Users\pc\rag-complaint-chatbot\.env')
groq_api_key = os.getenv('GROQ_API_KEY')

# ── Load everything once at startup ─────────────────────────
print("Loading vector store...")
vector_store_path = r'C:\Users\pc\rag-complaint-chatbot\vector_store'
index, df_chunks = load_vector_store(vector_store_path)

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Groq LLM...")
client = Groq(api_key=groq_api_key)

print("All systems ready!")


# ── Core functions ───────────────────────────────────────────

def build_prompt(question, retrieved_chunks):
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n[Complaint {i+1} - {chunk['product_category']}]\n"
        context += chunk['chunk_text']
        context += "\n"

    prompt = f"""You are a financial analyst assistant for CrediTrust Financial.
Your job is to analyze customer complaints and provide clear, insightful answers.
Use ONLY the complaint excerpts provided below to answer the question.
If the context does not contain enough information, say "I don't have enough information to answer this."
Do not make up information that is not in the complaints.

COMPLAINT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt


def generate_answer(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst assistant for CrediTrust Financial. Answer questions based only on the provided customer complaint excerpts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


def format_sources(retrieved_chunks):
    sources_text = ""
    icons = ["◆", "◇", "◈"]
    for i, chunk in enumerate(retrieved_chunks[:3]):
        sources_text += f"### {icons[i]} Source {i+1}\n"
        sources_text += f"**Product:** {chunk['product_category']}  \n"
        sources_text += f"**Company:** {chunk['company']}  \n"
        sources_text += f"**Issue:** {chunk['issue']}  \n\n"
        sources_text += f"> {chunk['chunk_text'][:180]}...\n\n"
        sources_text += "---\n"
    return sources_text


def chat(question, product_filter, history):
    if not question.strip():
        return history, "Ask a question to see sources here.", ""

    filter_value = None if product_filter == "All Products" else product_filter

    try:
        retrieved = retrieve(
            query=question,
            model=model,
            index=index,
            df_chunks=df_chunks,
            k=5,
            product_filter=filter_value
        )

        prompt = build_prompt(question, retrieved)
        answer = generate_answer(prompt)
        sources = format_sources(retrieved)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return history, sources, ""

    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"Something went wrong: {e}"})
        return history, "", ""


def clear_chat():
    return [], "Ask a question to see sources here.", ""


# ── Custom CSS — Deep Teal & Gold ────────────────────────────
custom_css = """

/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --teal-dark:   #0d3d3d;
    --teal-mid:    #155f5f;
    --teal-light:  #1e8080;
    --teal-pale:   #e8f4f4;
    --gold-dark:   #8a6500;
    --gold-mid:    #c49a00;
    --gold-light:  #f0c040;
    --gold-pale:   #fdf8e8;
    --cream:       #fafaf7;
    --text-dark:   #1a2e2e;
    --text-mid:    #4a6060;
    --text-light:  #8aabab;
    --border:      #d0e8e8;
    --shadow:      rgba(13, 61, 61, 0.12);
}

/* ── Base ── */
* { font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif !important; }

.gradio-container {
    background: var(--cream) !important;
    max-width: 1400px !important;
}

/* ── Split Header ── */
.split-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--teal-dark);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.split-header::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: rgba(240, 192, 64, 0.08);
    pointer-events: none;
}

.split-header::after {
    content: '';
    position: absolute;
    bottom: -60px;
    right: 80px;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: rgba(240, 192, 64, 0.05);
    pointer-events: none;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 18px;
}

.header-logo {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, var(--gold-light), var(--gold-mid));
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    box-shadow: 0 4px 16px rgba(240, 192, 64, 0.4);
    flex-shrink: 0;
}

.header-title {
    color: white;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin: 0;
    line-height: 1.2;
}

.header-subtitle {
    color: var(--gold-light);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 4px 0 0 0;
}

.header-right {
    text-align: right;
    max-width: 340px;
}

.header-tagline {
    color: rgba(255,255,255,0.75);
    font-size: 14px;
    font-weight: 400;
    line-height: 1.6;
    margin: 0 0 12px 0;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(240, 192, 64, 0.15);
    border: 1px solid rgba(240, 192, 64, 0.3);
    border-radius: 20px;
    padding: 5px 14px;
    color: var(--gold-light);
    font-size: 12px;
    font-weight: 600;
}

/* ── Stats Row ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}

.stat-card {
    background: white;
    border-radius: 14px;
    padding: 16px 20px;
    border: 1px solid var(--border);
    border-left: 4px solid var(--teal-light);
    box-shadow: 0 2px 12px var(--shadow);
    transition: transform 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
}

.stat-card:nth-child(2) { border-left-color: var(--gold-mid); }
.stat-card:nth-child(3) { border-left-color: var(--teal-mid); }
.stat-card:nth-child(4) { border-left-color: var(--gold-light); }

.stat-value {
    font-size: 22px;
    font-weight: 800;
    color: var(--teal-dark);
    line-height: 1;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 11px;
    color: var(--text-light);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Panel cards ── */
.panel-card {
    background: white;
    border-radius: 18px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 20px var(--shadow);
    overflow: hidden;
}

.panel-header {
    background: linear-gradient(135deg, var(--teal-dark), var(--teal-mid));
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.panel-header-title {
    color: white;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.panel-header-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--gold-light);
}

.source-panel-header {
    background: linear-gradient(135deg, var(--gold-dark), var(--gold-mid));
}

/* ── Chatbot ── */
.chat-wrap .wrap {
    border: none !important;
    border-radius: 0 !important;
    padding: 16px !important;
    background: white !important;
}

/* ── Input ── */
.input-row textarea,
.input-row input {
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--teal-pale) !important;
    color: var(--text-dark) !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s ease !important;
}

.input-row textarea:focus,
.input-row input:focus {
    border-color: var(--teal-light) !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(30, 128, 128, 0.1) !important;
    outline: none !important;
}

/* ── Ask button ── */
button.primary {
    background: linear-gradient(135deg, var(--teal-mid), var(--teal-dark)) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    padding: 12px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(13, 61, 61, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(13, 61, 61, 0.4) !important;
}

button.primary:active {
    transform: translateY(0px) !important;
}

/* ── Clear button ── */
button.secondary {
    background: white !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-mid) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover {
    border-color: var(--teal-light) !important;
    color: var(--teal-mid) !important;
    background: var(--teal-pale) !important;
}

/* ── Dropdown ── */
select, .dropdown {
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    background: white !important;
    color: var(--text-dark) !important;
    font-size: 13px !important;
}

/* ── Examples ── */
.examples-table table {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

.examples-table td {
    color: var(--teal-mid) !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
}

.examples-table tr:hover td {
    background: var(--teal-pale) !important;
}

/* ── Source markdown ── */
.source-md h3 {
    color: var(--teal-dark) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
}

.source-md blockquote {
    border-left: 3px solid var(--gold-mid) !important;
    background: var(--gold-pale) !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 10px 14px !important;
    margin: 8px 0 !important;
    font-size: 12px !important;
    color: var(--text-mid) !important;
    font-style: italic !important;
}

/* ── How it works ── */
.how-it-works {
    background: white;
    border-radius: 16px;
    border: 1px solid var(--border);
    padding: 24px;
    margin-top: 20px;
}

.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 16px;
}

.step-item {
    text-align: center;
    padding: 16px 12px;
    border-radius: 12px;
    background: var(--teal-pale);
    border: 1px solid var(--border);
    transition: transform 0.2s ease;
}

.step-item:hover { transform: translateY(-3px); }

.step-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.step-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--teal-dark);
    margin-bottom: 4px;
}

.step-desc {
    font-size: 11px;
    color: var(--text-light);
    line-height: 1.5;
}

.step-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--gold-mid);
    font-size: 20px;
    font-weight: 700;
}

/* ── Footer ── */
.footer-bar {
    background: var(--teal-dark);
    border-radius: 14px;
    padding: 16px 24px;
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-left {
    color: rgba(255,255,255,0.7);
    font-size: 13px;
}

.footer-left strong {
    color: var(--gold-light);
}

.footer-right {
    color: rgba(255,255,255,0.5);
    font-size: 12px;
}

/* ── Label hiding ── */
.hide-label label { display: none !important; }
"""

# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(css=custom_css) as app:

    # ── Split Header ─────────────────────────────────────────
    gr.HTML("""
    <div class="split-header">
        <div class="header-left">
            <div class="header-logo">🏛️</div>
            <div>
                <div class="header-title">CrediTrust Financial</div>
                <div class="header-subtitle">Complaint Intelligence Platform</div>
            </div>
        </div>
        <div class="header-right">
            <p class="header-tagline">
                Ask any question about customer complaints and get 
                instant, evidence-backed insights from real data.
            </p>
            <span class="header-badge">
                ◆ RAG · FAISS · Llama 3.3
            </span>
        </div>
    </div>
    """)

    # ── Stats Row ────────────────────────────────────────────
    gr.HTML("""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">477K+</div>
            <div class="stat-label">Complaints Processed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">29,024</div>
            <div class="stat-label">Searchable Chunks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">384-dim</div>
            <div class="stat-label">Embedding Vectors</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">4</div>
            <div class="stat-label">Product Categories</div>
        </div>
    </div>
    """)

    # ── Main Layout ──────────────────────────────────────────
    with gr.Row(equal_height=True):

        # ── Chat Panel ───────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML("""
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-header-dot"></div>
                    <div class="panel-header-title">Complaint Analysis Chat</div>
                </div>
            </div>
            """)

            with gr.Group(elem_classes="panel-card chat-wrap"):
                chatbot = gr.Chatbot(
                    label="",
                    height=380,
                    show_label=False,
                    elem_classes="chat-wrap"
                )

                with gr.Row(elem_classes="input-row"):
                    question_input = gr.Textbox(
                        placeholder="Ask about customer complaints...",
                        label="",
                        show_label=False,
                        scale=5,
                        lines=1,
                        elem_classes="hide-label"
                    )
                    submit_btn = gr.Button(
                        "Ask →",
                        variant="primary",
                        scale=1
                    )

                with gr.Row():
                    product_filter = gr.Dropdown(
                        choices=[
                            "All Products",
                            "Credit Card",
                            "Savings Account",
                            "Money Transfer",
                            "Personal Loan"
                        ],
                        value="All Products",
                        label="Filter by Product",
                        scale=3
                    )
                    clear_btn = gr.Button(
                        "Clear ✕",
                        variant="secondary",
                        scale=1
                    )

                gr.HTML('<div style="font-size:12px;font-weight:600;color:#8aabab;margin:12px 0 6px;letter-spacing:0.8px;text-transform:uppercase;">Try these →</div>')
                gr.Examples(
                    examples=[
                        ["Why are people unhappy with credit cards?", "All Products"],
                        ["What fraud issues are reported in money transfers?", "Money Transfer"],
                        ["What are the main issues with savings accounts?", "Savings Account"],
                        ["Why do people complain about personal loans?", "Personal Loan"],
                        ["How do companies respond to complaints?", "All Products"],
                    ],
                    inputs=[question_input, product_filter],
                    label=""
                )

        # ── Sources Panel ────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML("""
            <div class="panel-card">
                <div class="panel-header source-panel-header">
                    <div class="panel-header-dot" style="background:#0d3d3d;"></div>
                    <div class="panel-header-title">Evidence Sources</div>
                </div>
            </div>
            """)

            with gr.Group(elem_classes="panel-card"):
                gr.HTML("""
                <div style="padding:12px 16px 0;font-size:12px;color:#8aabab;
                            border-bottom:1px solid #d0e8e8;padding-bottom:12px;
                            line-height:1.5;">
                    Every answer is grounded in <strong style="color:#155f5f;">
                    real complaint data</strong>. Sources are shown here 
                    so you can verify and trace each insight.
                </div>
                """)
                sources_output = gr.Markdown(
                    value="\n\n*Ask a question to see which complaints were used.*",
                    elem_classes="source-md",
                    container=False
                )

    # ── How it works ─────────────────────────────────────────
    gr.HTML("""
    <div class="how-it-works">
        <div style="font-size:13px;font-weight:700;color:#8aabab;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">
            Under the Hood
        </div>
        <div style="font-size:18px;font-weight:800;color:#0d3d3d;">
            How Your Answer is Generated
        </div>
        <div style="display:grid;grid-template-columns:1fr 24px 1fr 24px 1fr 24px 1fr;
                    gap:8px;margin-top:16px;align-items:center;">
            <div class="step-item">
                <div class="step-icon">✏️</div>
                <div class="step-title">You Ask</div>
                <div class="step-desc">Type any question in plain English</div>
            </div>
            <div style="text-align:center;color:#c49a00;font-size:18px;font-weight:700;">→</div>
            <div class="step-item">
                <div class="step-icon">🔢</div>
                <div class="step-title">Embed</div>
                <div class="step-desc">MiniLM converts your question to a 384-dim vector</div>
            </div>
            <div style="text-align:center;color:#c49a00;font-size:18px;font-weight:700;">→</div>
            <div class="step-item">
                <div class="step-icon">🔍</div>
                <div class="step-title">Retrieve</div>
                <div class="step-desc">FAISS searches 29K chunks for the closest matches</div>
            </div>
            <div style="text-align:center;color:#c49a00;font-size:18px;font-weight:700;">→</div>
            <div class="step-item">
                <div class="step-icon">🧠</div>
                <div class="step-title">Generate</div>
                <div class="step-desc">Llama 3.3 synthesizes a grounded answer</div>
            </div>
        </div>
    </div>
    """)

    # ── Footer ───────────────────────────────────────────────
    gr.HTML("""
    <div class="footer-bar">
        <div class="footer-left">
            Built by <strong>Rebika Woldeyesus</strong> — 
            Kifiya AI Training Program · 10 Academy · Addis Ababa 🇪🇹
        </div>
        <div class="footer-right">
            Week 7 · RAG · FAISS · Llama 3.3 · Gradio
        </div>
    </div>
    """)

    # ── Event Handlers ───────────────────────────────────────
    submit_btn.click(
        fn=chat,
        inputs=[question_input, product_filter, chatbot],
        outputs=[chatbot, sources_output, question_input]
    )

    question_input.submit(
        fn=chat,
        inputs=[question_input, product_filter, chatbot],
        outputs=[chatbot, sources_output, question_input]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, sources_output, question_input]
    )


# ── Launch ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        share=False,
        server_port=7860,
        show_error=True
    )