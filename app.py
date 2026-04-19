import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Page Config ---
st.set_page_config(page_title="AI Research Assistant", page_icon="📄", layout="wide")

# --- API Key Logic ---
# Pehle Secrets check karega, agar nahi mili to sidebar se input lega
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.sidebar.warning("🔑 Groq API Key missing!")
    api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
    if not api_key:
        st.info("Please add your API Key in Streamlit Secrets or Sidebar to continue.")
        st.stop()

# --- Initialize Groq LLM ---
try:
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="mixtral-8x7b-32768",
        temperature=0.5
    )
except Exception as e:
    st.error(f"LLM Initialization Error: {e}")
    st.stop()

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        st.error(f"PDF Read Error: {e}")
        return None

# --- UI ---
st.title("📄 AI Research Paper Summarizer & Rewriter")
st.markdown("Upload a PDF to get a smart summary or plagiarism-free rewrite.")

uploaded_file = st.file_uploader("Upload your Research Paper", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF content..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    
    if raw_text:
        # Text ko chotay chunks mein divide karna taake BadRequestError na aaye
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        # Sirf pehle kuch chunks use karte hain context ke liye (Token limit se bachne ke liye)
        context_text = "\n".join(chunks[:3]) 

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Summarize Paper"):
                with st.spinner("Generating Summary..."):
                    try:
                        prompt = f"Provide a section-wise summary (Abstract, Methodology, Results, Conclusion) for this research paper:\n\n{context_text}"
                        response = llm.invoke(prompt)
                        st.subheader("Summary")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Summarization Error: {e}")

        with col2:
            if st.button("✨ Plagiarism-Free Rewrite"):
                with st.spinner("Rewriting Content..."):
                    try:
                        prompt = f"Rewrite the following research content to be plagiarism-free while keeping scientific accuracy:\n\n{context_text}"
                        response = llm.invoke(prompt)
                        st.subheader("Rewritten Text")
                        st.write(response.content)
                    except Exception as e:
                        st.error(f"Rewriting Error: {e}")
    else:
        st.error("PDF se text extract nahi ho saka. File check karein.")

# --- README Section (Jab ready ho jaye tab use karein) ---
with st.expander("How to use this tool?"):
    st.write("""
    1. Upload your research PDF.
    2. Click 'Summarize' to get the core idea.
    3. Click 'Rewrite' to get a unique version of the text.
    4. Make sure your API key has enough quota!
    """)
