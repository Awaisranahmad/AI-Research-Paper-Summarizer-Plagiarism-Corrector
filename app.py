import streamlit as st
import os
import re
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="AI Research Paper Assistant — Expert",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Premium Look)
st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #0f172a 0%, #1e2937 100%); color: #e2e8f0; }
    .stButton>button { height: 52px; font-weight: 600; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 10px 24px; }
    .metric-card { background: rgba(255,255,255,0.08); border-radius: 16px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# ====================== API KEY ======================
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("🚨 GROQ_API_KEY missing! Add it in Streamlit Secrets.")
    st.stop()

# ====================== SIDEBAR — PRO SETTINGS ======================
with st.sidebar:
    st.title("⚙️ Expert Controls")
    st.caption("Real-time settings • Groq powered")
    
    model_options = {
        "Llama 3.3 70B (Best)": "llama-3.3-70b-versatile",
        "Llama 3 70B": "llama3-70b-8192",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma2 9B": "gemma2-9b-it"
    }
    selected_model = st.selectbox("Model", options=list(model_options.keys()), index=0)
    model_name = model_options[selected_model]
    
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.slider("Max Output Tokens", 512, 8192, 2048, step=256)
    chunk_size = st.slider("Chunk Size (characters)", 2000, 6000, 3500, step=500)
    chunk_overlap = st.slider("Chunk Overlap", 200, 800, 400, step=100)
    
    st.divider()
    st.markdown("**Advanced Options**")
    use_full_context = st.toggle("Use full document context (slower but better)", value=True)
    auto_analyze = st.toggle("Auto-run metadata + key findings", value=True)
    
    st.caption(f"🔥 Using <b>{selected_model}</b> • {datetime.now().strftime('%H:%M')}", unsafe_allow_html=True)

# Initialize LLM
@st.cache_resource(show_spinner=False)
def get_llm(_model_name, _temp, _max_tokens):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=_model_name,
        temperature=_temp,
        max_tokens=_max_tokens,
        top_p=0.95,
        streaming=False
    )

llm = get_llm(model_name, temperature, max_tokens)

# ====================== HELPER FUNCTIONS ======================
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_metadata(text_chunk):
    prompt = f"""Extract the following information from this research paper in clean Markdown format.
Only use the text provided. If something is missing, write "Not found".

- **Title**:
- **Authors**:
- **Year**:
- **Journal / Conference**:
- **DOI** (if any):
- **Keywords** (5-8):
- **Abstract** (first 2-3 sentences):

Paper text:
{text_chunk[:12000]}"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except:
        return "⚠️ Metadata extraction failed."

def generate_structured_analysis(context, task):
    prompts = {
        "key_findings": "Extract the 8 most important key findings/results in bullet points with page/section reference if possible.",
        "limitations": "List all mentioned limitations, weaknesses, and future work suggestions.",
        "methodology": "Summarize the research methodology, data, and tools used in detail.",
        "gaps": "Identify research gaps and potential future research directions.",
        "critique": "Give a professional academic critique: strengths, weaknesses, novelty, and impact."
    }
    prompt = f"""{prompts[task]}
Be concise, professional, and evidence-based.
Context:
{context[:28000]}"""
    try:
        res = llm.invoke(prompt)
        return res.content
    except Exception as e:
        return f"Error: {e}"

# ====================== MAIN UI ======================
st.title("🚀 AI Research Paper Assistant")
st.markdown("**Expert Edition v2.0** — Llama 3.3 + Advanced RAG + Pro UI")
st.caption("Upload any research PDF • Get metadata, summaries, deep analysis & chat instantly")

uploaded_file = st.file_uploader("📤 Upload Research Paper (PDF)", type="pdf", label_visibility="collapsed")

if uploaded_file:
    # Process PDF only once
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("🔥 Extracting & chunking full paper..."):
            reader = PdfReader(uploaded_file)
            raw_text = "".join([page.extract_text() or "" for page in reader.pages])
            cleaned_text = clean_text(raw_text)
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(cleaned_text)
            
            full_context = "\n\n───\n\n".join(chunks[:12]) if not use_full_context else cleaned_text[:32000]
            
            st.session_state.processed_file = uploaded_file.name
            st.session_state.cleaned_text = cleaned_text
            st.session_state.chunks = chunks
            st.session_state.full_context = full_context
            st.session_state.num_pages = len(reader.pages)
            st.session_state.metadata = None
            st.session_state.chat_history = []
            
            st.success(f"✅ Processed **{len(chunks)} chunks** • {len(reader.pages)} pages • {len(cleaned_text):,} characters")

    # Tabs
    tab_overview, tab_summary, tab_analysis, tab_chat, tab_rewrite, tab_export = st.tabs([
        "📊 Overview", "📝 Summaries", "🔬 Deep Analysis", "💬 Ask Anything", "✍️ Rewrite", "📤 Export"
    ])

    context = st.session_state.full_context
    cleaned = st.session_state.cleaned_text

    # TAB 1: Overview
    with tab_overview:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("📋 Document Stats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages", st.session_state.num_pages)
            c2.metric("Chunks", len(st.session_state.chunks))
            c3.metric("Words", f"{len(cleaned.split()):,}")
            c4.metric("Characters", f"{len(cleaned):,}")
            st.subheader("🔍 First Page Preview")
            st.text_area("", cleaned[:1500], height=220, disabled=True)
        
        with col2:
            st.subheader("📌 Extract Metadata")
            if st.button("🔥 Extract Title, Authors, DOI & Keywords", use_container_width=True, type="primary") or st.session_state.metadata is None:
                with st.spinner("Analyzing..."):
                    st.session_state.metadata = extract_metadata(cleaned)
            if st.session_state.metadata:
                st.markdown(st.session_state.metadata)

    # TAB 2: Summaries
    with tab_summary:
        summary_style = st.selectbox("Choose Summary Style", [
            "Concise (300 words)", "Detailed Academic", "Executive Summary", 
            "One-Page Abstract Style", "Layman Explanation (for non-experts)"
        ])
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            with st.spinner("Llama 3.3 thinking..."):
                prompt = f"Write a **{summary_style}** of the research paper. Use professional tone, highlight contribution and results.\n\nContext:\n{context}"
                result = llm.invoke(prompt)
                st.markdown("### 📄 Generated Summary")
                st.write(result.content)
                st.download_button("⬇️ Download as .txt", result.content, f"summary_{uploaded_file.name}.txt")

    # TAB 3: Deep Analysis
    with tab_analysis:
        st.subheader("Choose Analysis Type")
        cols = st.columns(3)
        analysis_types = [
            ("🔑 Key Findings", "key_findings"),
            ("⚠️ Limitations & Future Work", "limitations"),
            ("🔬 Methodology", "methodology"),
            ("🌍 Research Gaps", "gaps"),
            ("📈 Full Academic Critique", "critique")
        ]
        for i, (label, key) in enumerate(analysis_types):
            with cols[i % 3]:
                if st.button(label, use_container_width=True):
                    with st.spinner("Generating deep analysis..."):
                        result = generate_structured_analysis(context, key)
                        st.markdown(f"### {label}")
                        st.write(result)

    # TAB 4: Q&A Chat
    with tab_chat:
        st.markdown("**💬 Ask anything about this paper** (RAG-powered)")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Example: What was the main research question? How did they collect data?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Thinking..."):
                qa_prompt = f"""You are an expert researcher. Answer ONLY using the provided paper context.
If the answer is not in the paper, say "Not mentioned in this paper."

Paper Context:
{context[:30000]}

Question: {prompt}"""
                response = llm.invoke(qa_prompt)
                answer = response.content
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

    # TAB 5: Rewrite
    with tab_rewrite:
        rewrite_mode = st.selectbox("Rewrite Style", [
            "Plagiarism-free Academic Version",
            "Simplified for Students / Blog",
            "Journal-Ready (Nature/Science style)",
            "Conference Presentation Version",
            "Twitter Thread Style (10 tweets)"
        ])
        length = st.radio("Length", ["Keep similar length", "Make it shorter", "Make it longer"], horizontal=True)
        
        if st.button("✨ Rewrite Full Paper Section", type="primary", use_container_width=True):
            with st.spinner("Rewriting..."):
                prompt = f"""Rewrite the following research text in **{rewrite_mode}**.
Make it 100% original, improve clarity and flow.
Length instruction: {length}

Original text:
{context[:25000]}"""
                result = llm.invoke(prompt)
                st.write(result.content)
                st.download_button("Download Rewritten Version", result.content, "rewritten_paper.md")

    # TAB 6: Export
    with tab_export:
        st.subheader("Download Everything")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Full Cleaned Text", st.session_state.cleaned_text, f"cleaned_{uploaded_file.name}.txt")
            st.download_button("📋 All Metadata", st.session_state.metadata or "No metadata yet", "metadata.md")
        with col2:
            if st.button("📦 Generate Complete Research Report"):
                with st.spinner("Creating full report..."):
                    report_prompt = f"""Create a complete professional research report with these sections:
1. Title & Authors
2. Executive Summary
3. Key Findings
4. Methodology
5. Limitations & Gaps
6. Academic Critique

Use the paper below:
{context[:32000]}"""
                    report = llm.invoke(report_prompt).content
                    st.success("✅ Full report generated!")
                    st.download_button("⬇️ Download Complete Report", report, "FULL_RESEARCH_REPORT.md", type="primary")

else:
    st.info("👆 Upload a PDF to unlock all expert features", icon="📄")
    st.markdown("""
    ### 🔥 Expert Features Included:
    - 6 Modern Tabs
    - Auto Metadata Extractor
    - Unlimited Q&A Chat
    - Deep Analysis (Key Findings, Gaps, Critique etc.)
    - 5 Rewrite Styles
    - Full Research Report Generator
    - Pro Sidebar Settings
    """)

st.caption("Made with ❤️ for researchers • Powered by Groq + Llama 3.3 70B")
