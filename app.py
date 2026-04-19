import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq

# Naya Import Style
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Page Setup ---
st.set_page_config(page_title="AI Research Assistant", layout="wide", page_icon="📄")

# --- API Key Logic ---
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Missing API Key! Please add it to Streamlit Secrets or .env file.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768")

# --- UI Layout ---
st.title("📄 AI Research Paper Assistant")
st.write("Upload a PDF to summarize or rewrite it without plagiarism.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # 1. Text Extraction
    with st.spinner("Extracting text from PDF..."):
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

    if raw_text:
        # 2. Text Splitting (Important for long papers)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        chunks = text_splitter.split_text(raw_text)
        context = chunks[0]  # First 3000 characters for testing

        # 3. Action Buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Summarize"):
                with st.spinner("Summarizing..."):
                    res = llm.invoke(f"Provide a clear, section-wise summary of this research text:\n\n{context}")
                    st.subheader("Summary")
                    st.write(res.content) # Yahan use ho rahi hai wo line

        with col2:
            if st.button("✨ Rewrite (No Plagiarism)"):
                with st.spinner("Rewriting..."):
                    res = llm.invoke(f"Rewrite the following text to be plagiarism-free and unique while maintaining academic accuracy:\n\n{context}")
                    st.subheader("Plagiarism-Free Content")
                    st.write(res.content)
    else:
        st.error("Could not read the PDF. Please try another file.")
