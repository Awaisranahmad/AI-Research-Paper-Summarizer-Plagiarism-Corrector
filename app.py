import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq

# 1. API Key Load karna
load_dotenv()
# Agar .env file mein hai to wahan se uthayega, 
# warna Streamlit Cloud ke 'Secrets' se check karega
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# UI Setup
st.set_page_config(page_title="AI Research Assistant", page_icon="📄")
st.title("📄 AI Paper Summarizer & Rewriter")

if not api_key:
    st.error("API Key nahi mili! Please .env file check karein ya Streamlit Secrets mein add karein.")
    st.stop()

# Initialize LLM (Groq)
llm = ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768")

def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# --- Sidebar / File Upload ---
uploaded_file = st.file_uploader("Upload Research Paper", type="pdf")

if uploaded_file:
    raw_text = process_pdf(uploaded_file)
    # Text ko thora chota rakhte hain taake LLM crash na ho
    context = raw_text[:8000] 

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Summarize Paper"):
            with st.spinner("Summarizing..."):
                prompt = f"Summarize this research paper section-wise (Abstract, Methodology, Results, Conclusion):\n\n{context}"
                response = llm.invoke(prompt)
                st.markdown("### 📝 Section-wise Summary")
                st.write(response.content)

    with col2:
        if st.button("Plagiarism-Free Rewrite"):
            with st.spinner("Rewriting..."):
                prompt = f"Rewrite the following text to be 100% plagiarism-free but keep the academic meaning intact:\n\n{context}"
                response = llm.invoke(prompt)
                st.markdown("### ✨ Rewritten Content")
                st.write(response.content)
