import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-2.5-flash')

@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.set_page_config(page_title='RAGCHAT AI', page_icon='⚡', layout='centered')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #07070E !important;
    color: #E2E2F0 !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 680px !important;
    padding: 0 24px 80px !important;
}

/* ── NAV ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 0 28px;
    border-bottom: 1px solid #13132A;
    margin-bottom: 32px;
}
.nav-brand { display:flex; align-items:center; gap:10px; }
.nav-logo {
    width:32px; height:32px;
    background: linear-gradient(135deg,#7C6FFF,#FF5FA0);
    border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    font-size:15px; font-weight:700; color:#fff;
    font-family:'Space Mono',monospace;
}
.nav-name { font-size:15px; font-weight:700; color:#F0F0FF; }
.nav-name em {
    font-style:normal;
    background:linear-gradient(135deg,#7C6FFF,#FF5FA0);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.nav-pill {
    display:flex; align-items:center; gap:6px;
    background:#0F0F1E; border:1px solid #1E1E38;
    border-radius:20px; padding:5px 12px;
    font-size:11px; font-weight:600; color:#5A5A8A;
    font-family:'Space Mono',monospace;
}
.live-dot {
    width:6px; height:6px; border-radius:50%;
    background:#22C55E;
    animation:blink 2s infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── HERO ── */
.hero { text-align:center; padding:4px 0 30px; }
.eyebrow {
    display:inline-flex; align-items:center; gap:7px;
    background:#0F0F1E; border:1px solid #252548;
    border-radius:20px; padding:5px 14px;
    font-size:10px; font-weight:700;
    letter-spacing:1.4px; text-transform:uppercase;
    color:#7C6FFF; font-family:'Space Mono',monospace;
    margin-bottom:18px;
}
.e-dot { width:5px;height:5px;border-radius:50%;background:#7C6FFF;animation:blink 1.6s infinite; }
.hero h1 {
    font-size:40px; font-weight:700;
    line-height:1.1; letter-spacing:-1.5px;
    color:#F0F0FF; margin-bottom:12px;
}
.grad {
    background:linear-gradient(135deg,#7C6FFF 0%,#FF5FA0 60%,#FF9A3C 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.tagline {
    font-size:12px; color:#2E2E50;
    font-weight:700; letter-spacing:4px;
    text-transform:uppercase;
    font-family:'Space Mono',monospace;
    margin-bottom:26px;
}
.tagline em { font-style:normal; color:#7C6FFF; }

/* ── PIPELINE — fixed width chips, wrap on small screens ── */
.pipeline {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:6px;
    margin-bottom:36px;
    overflow-x:auto;
    padding-bottom:4px;
}
.pipeline::-webkit-scrollbar { height:3px; }
.pipeline::-webkit-scrollbar-thumb { background:#1E1E38; border-radius:4px; }

.p-step {
    display:inline-flex;
    align-items:center;
    gap:5px;
    background:#0F0F1E;
    border:1px solid #1E1E38;
    border-radius:8px;
    padding:6px 10px;
    font-size:11px;
    font-weight:600;
    color:#505080;
    font-family:'Space Mono',monospace;
    white-space:nowrap;
    flex-shrink:0;
}
.p-step.active { border-color:#7C6FFF; color:#9080FF; background:#12122A; }
.p-arrow { color:#1E1E38; font-size:12px; flex-shrink:0; }

/* ── SECTION LABEL ── */
.sec-label {
    display:flex; align-items:center; gap:8px;
    font-size:10px; font-weight:700;
    letter-spacing:1px; text-transform:uppercase;
    color:#3A3A60; font-family:'Space Mono',monospace;
    margin-bottom:10px;
}
.s-badge {
    width:20px; height:20px;
    background:linear-gradient(135deg,#7C6FFF,#FF5FA0);
    border-radius:6px;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:10px; font-weight:700; color:#fff;
}

/* ── FILE UPLOADER — hide duplicate text bug ── */
[data-testid="stFileUploaderDropzone"] {
    background:#0C0C1A !important;
    border:1.5px dashed #252548 !important;
    border-radius:12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color:#7C6FFF !important;
    background:#0F0F22 !important;
}
/* hide ALL text/spans inside dropzone to prevent duplication */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small { display:none !important; }
/* keep only the button visible */
[data-testid="stFileUploaderDropzone"] button {
    display:flex !important;
    background:#13132A !important;
    border:1px solid #2A2A50 !important;
    color:#7C6FFF !important;
    border-radius:8px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-size:13px !important;
    font-weight:600 !important;
    padding:8px 20px !important;
    margin:16px auto !important;
}
[data-testid="stFileUploaderDropzone"]::after {
    content:"Drop PDF here or click to browse";
    display:block;
    text-align:center;
    font-size:13px;
    color:#3A3A60;
    padding-bottom:14px;
    font-family:'Space Grotesk',sans-serif;
}

/* ── METRICS ── */
.metrics {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:8px;
    margin:14px 0 20px;
}
.metric {
    background:#0C0C1A;
    border:1px solid #1A1A35;
    border-radius:12px;
    padding:14px 10px;
    text-align:center;
}
.m-val {
    font-size:20px; font-weight:700;
    font-family:'Space Mono',monospace;
    background:linear-gradient(135deg,#7C6FFF,#FF5FA0);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    letter-spacing:-0.5px;
}
.m-key {
    font-size:9px; font-weight:700;
    letter-spacing:.8px; text-transform:uppercase;
    color:#2E2E50; margin-top:4px;
    font-family:'Space Mono',monospace;
}

/* ── BANNERS ── */
.banner {
    display:flex; align-items:center; gap:10px;
    padding:12px 16px; border-radius:10px;
    font-size:13px; font-weight:500;
    margin:10px 0; line-height:1.5;
}
.ok   { background:#071F14; border:1px solid #0F4030; color:#34D399; }
.warn { background:#1A100A; border:1px solid #4A2A10; color:#FB923C; }
.info { background:#0C0C1A; border:1px solid #2A2A50; color:#7C6FFF; }

.divider {
    height:1px;
    background:linear-gradient(90deg,transparent,#1A1A35 30%,#1A1A35 70%,transparent);
    margin:22px 0;
}

/* ── TEXT INPUT ── */
.stTextInput input {
    font-family:'Space Grotesk',sans-serif !important;
    background:#0C0C1A !important;
    border:1.5px solid #252548 !important;
    border-radius:12px !important;
    color:#E2E2F0 !important;
    font-size:15px !important;
    padding:13px 16px !important;
    caret-color:#7C6FFF;
    transition:border-color .2s, box-shadow .2s;
}
.stTextInput input:focus {
    border-color:#7C6FFF !important;
    box-shadow:0 0 0 3px rgba(124,111,255,.12) !important;
}
.stTextInput input::placeholder { color:#252545 !important; }
.stTextInput label { display:none !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color:#7C6FFF !important; }
[data-testid="stSpinner"] p {
    color:#3A3A60 !important;
    font-size:11px !important;
    font-family:'Space Mono',monospace !important;
}

/* ── ANSWER ── */
.ans-wrap {
    background:#0C0C1A;
    border:1px solid #1E1E38;
    border-radius:14px;
    overflow:hidden;
    margin-top:16px;
    animation:fadeUp .35s ease;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.ans-top {
    background:#10102A;
    border-bottom:1px solid #1E1E38;
    padding:11px 18px;
    display:flex; align-items:center; justify-content:space-between;
}
.ans-tag {
    display:flex; align-items:center; gap:7px;
    font-size:10px; font-weight:700;
    letter-spacing:1.2px; text-transform:uppercase;
    color:#7C6FFF; font-family:'Space Mono',monospace;
}
.a-ping { width:7px;height:7px;border-radius:50%;background:#7C6FFF;animation:blink 2s infinite; }
.ans-model { font-size:10px; color:#252545; font-family:'Space Mono',monospace; }
.ans-body {
    padding:18px 20px;
    font-size:14.5px; line-height:1.85;
    color:#C0C0E0; white-space:pre-wrap;
}

/* ── FOOTER ── */
.footer {
    text-align:center; padding-top:36px;
    font-size:10px; color:#18183A;
    font-family:'Space Mono',monospace; letter-spacing:.5px;
}
.footer b {
    background:linear-gradient(135deg,#7C6FFF,#FF5FA0);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
</style>
""", unsafe_allow_html=True)

# ── NAV ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="nav-brand">
    <div class="nav-logo">R</div>
    <span class="nav-name"><em>RAGCHAT</em> AI</span>
  </div>
  <div class="nav-pill">
    <span class="live-dot"></span> Gemini 2.5 Flash · Live
  </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="eyebrow"><span class="e-dot"></span> Retrieval Augmented Generation</div>
  <h1>Chat with your<br><span class="grad">Documents.</span></h1>
  <div class="tagline">Upload<em>.</em> Ask<em>.</em> Understand<em>.</em></div>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
  <span class="p-step active">📄 Parse</span>
  <span class="p-arrow">→</span>
  <span class="p-step">✂️ Chunk</span>
  <span class="p-arrow">→</span>
  <span class="p-step">🔷 Embed</span>
  <span class="p-arrow">→</span>
  <span class="p-step">🔍 Retrieve</span>
  <span class="p-arrow">→</span>
  <span class="p-step">⚡ Generate</span>
</div>
""", unsafe_allow_html=True)

# ── LOAD EMBEDDING ───────────────────────────────────────────────────────────
with st.spinner('Loading embedding model…'):
    embedding_model = load_embedding()

# ── STEP 1: UPLOAD ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-label"><span class="s-badge">1</span> Upload your PDF</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader('', type=['pdf'], label_visibility='collapsed')

# ── PROCESS PDF ──────────────────────────────────────────────────────────────
raw_text = ''
if uploaded_file:
    pdf = PdfReader(uploaded_file)
    for page in pdf.pages:
        raw_text += page.extract_text() or ''

if raw_text.strip():
    doc = Document(page_content=raw_text)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([doc])
    texts = [c.page_content for c in chunks]
    vector_db = FAISS.from_texts(texts, embedding_model)
    retriever = vector_db.as_retriever()
    wc = len(raw_text.split())

    st.markdown(f"""
    <div class="banner ok">✅ &nbsp; Indexed successfully — ready for your questions</div>
    <div class="metrics">
      <div class="metric"><div class="m-val">{len(pdf.pages)}</div><div class="m-key">Pages</div></div>
      <div class="metric"><div class="m-val">{len(chunks)}</div><div class="m-key">Chunks</div></div>
      <div class="metric"><div class="m-val">{wc:,}</div><div class="m-key">Words</div></div>
      <div class="metric"><div class="m-val">768</div><div class="m-key">Dims</div></div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # ── STEP 2: QUERY ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label"><span class="s-badge">2</span> Ask your question</div>', unsafe_allow_html=True)

    query = st.text_input('q', placeholder='e.g.  What are the key skills of the candidate?', label_visibility='collapsed')

    if query:
        with st.spinner('Searching & generating answer…'):
            docs = retriever.invoke(query)
            context = '\n\n'.join([d.page_content for d in docs])
            prompt = f"""You are a precise document analyst for RAGCHAT AI.
Answer using only the context provided. Be clear and concise.
If the answer is not in the context, say: "I couldn't find that in the document."

Context:
{context}

Question: {query}
Answer:"""
            resp = model.generate_content(prompt)

        st.markdown(f"""
        <div class="ans-wrap">
          <div class="ans-top">
            <div class="ans-tag"><span class="a-ping"></span> RAGCHAT AI · Answer</div>
            <div class="ans-model">gemini-2.5-flash · MiniLM-L6-v2</div>
          </div>
          <div class="ans-body">{resp.text}</div>
        </div>
        """, unsafe_allow_html=True)

elif uploaded_file:
    st.markdown('<div class="banner warn">⚠️ &nbsp; Could not extract text — please use a text-based PDF, not a scanned image.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="banner info">💡 &nbsp; Drop a PDF above to get started. Supports resumes, research papers, contracts & more.</div>', unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="footer"><b>RAGCHAT AI</b> · Upload. Ask. Understand. · Gemini 2.5 Flash + FAISS</div>', unsafe_allow_html=True)
