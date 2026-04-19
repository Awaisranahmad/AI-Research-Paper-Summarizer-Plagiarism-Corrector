import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI # Ya Gemini/Groq jo aap use krna chahen

# App UI Configuration
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📄 AI Paper Summarizer & Plagiarism Corrector")

def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        raw_text = process_pdf(uploaded_file)
        
        # Text splitting for long papers
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Summarize Section-wise"):
            # Yahan Summarization Prompt jayega
            st.subheader("Summary")
            st.info("Generating summary for each section...")
            # LLM logic goes here
            
    with col2:
        if st.button("Correct Plagiarism (Rewrite)"):
            # Yahan Rewriting Logic jayega
            st.subheader("Plagiarism-Free Content")
            st.success("Rewriting content in your own words...")
