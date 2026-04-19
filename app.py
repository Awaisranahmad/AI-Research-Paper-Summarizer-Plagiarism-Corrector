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
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# --- API Key Logic ---
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("API Key missing! Please add it to Secrets.")
    st.stop()

# Initialize LLM with Safety
try:
    llm = ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768", temperature=0.3)
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- Functions ---
def clean_text(text):
    # Sirf kaam ka text rakhna (Remove non-printable characters)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- UI ---
st.title("📄 AI Research Paper Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        # Text Clean up
        cleaned_text = clean_text(raw_text)

    if cleaned_text:
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(cleaned_text)
        
        # Pehle chunk par process karte hain safety ke liye
        context = chunks[0]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Summarize"):
                with st.spinner("Talking to AI..."):
                    try:
                        # Simple and clean prompt
                        prompt = f"Summarize the following research text concisely:\n\n{context}"
                        res = llm.invoke(prompt)
                        st.subheader("Summary")
                        st.write(res.content)
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")

        with col2:
            if st.button("✨ Rewrite"):
                with st.spinner("Rewriting..."):
                    try:
                        prompt = f"Rewrite this text to be plagiarism-free and academic:\n\n{context}"
                        res = llm.invoke(prompt)
                        st.subheader("Rewritten Content")
                        st.write(res.content)
                    except Exception as e:
                        st.error(f"Error during rewriting: {e}")
    else:
        st.error("PDF is empty or unreadable.")
