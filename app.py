import streamlit as st
import os
import re
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq

# Import Fix
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Page Setup ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="📄")

# --- API Key Logic ---
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Missing API Key! Please add it to Secrets.")
    st.stop()

# Initialize LLM with Supported Model (Llama 3.3)
try:
    # 'mixtral' ki jagah 'llama-3.3-70b-versatile' use kar rahe hain
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.3
    )
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- Functions ---
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- UI ---
st.title("📄 AI Research Paper Assistant")
st.write("Using Llama 3.3 for high-speed summarization.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        cleaned_text = clean_text(raw_text)

    if cleaned_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = text_splitter.split_text(cleaned_text)
        
        context = chunks[0] # First chunk

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Summarize"):
                with st.spinner("Analyzing..."):
                    try:
                        prompt = f"Provide a concise, professional summary of this research text:\n\n{context}"
                        res = llm.invoke(prompt)
                        st.subheader("Summary")
                        st.write(res.content)
                    except Exception as e:
                        st.error(f"Summarization Error: {e}")

        with col2:
            if st.button("✨ Rewrite"):
                with st.spinner("Rewriting..."):
                    try:
                        prompt = f"Rewrite this text to be plagiarism-free and academic:\n\n{context}"
                        res = llm.invoke(prompt)
                        st.subheader("Rewritten Content")
                        st.write(res.content)
                    except Exception as e:
                        st.error(f"Rewriting Error: {e}")
    else:
        st.error("PDF is empty or unreadable.")
