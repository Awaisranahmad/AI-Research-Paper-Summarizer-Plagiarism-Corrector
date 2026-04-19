🚀 AI Research Paper Assistant — Expert Edition v2.0
An advanced, high-performance academic research tool built with Streamlit, LangChain, and Groq (Llama 3.3 70B). This assistant is designed for students and researchers to perform deep analysis, summarization, and plagiarism-free rewriting of complex research papers in seconds.

🌟 Key Features
📊 1. Intelligent Overview & Metadata
Auto-Extraction: Instantly identifies Title, Authors, Year, DOI, and Keywords.

Document Stats: Real-time metrics on page count, word count, and character density.

Smart Chunking: Uses RecursiveCharacterTextSplitter to maintain context across large PDFs.

📝 2. Multi-Style Summaries
Choose from 5 different summary styles:

Concise (300 words)

Detailed Academic

Executive Summary

One-Page Abstract

Layman Explanation

🔬 3. Deep Academic Analysis
Key Findings: Automatically extracts the 8 most critical results.

Methodology: Detailed breakdown of research tools and data.

Limitations & Gaps: Identifies research weaknesses and future directions.

Academic Critique: Provides professional feedback on novelty and impact.

💬 4. RAG-Powered Expert Chat
An interactive chatbot that answers questions strictly based on the paper's context.

Maintains chat history for a seamless research experience.

✍️ 5. AI Rewriter (Plagiarism-Free)
Transform text into 5 different formats:

Journal-Ready (Nature/Science style)

Student/Blog version

Twitter Thread (10 tweets)

Conference Presentation

📤 6. Professional Export
Generate a Complete Research Report covering all sections.

Download summaries and rewritten versions as .txt or .md files.

🛠️ Technical Stack
UI Framework: Streamlit (Custom Premium CSS)

LLM Engine: Groq LPU (Llama 3.3 70B, Llama 3 70B, Mixtral)

Orchestration: LangChain

PDF Engine: PyPDF2

Text Processing: Regex + Recursive Chunking

⚙️ Installation & Setup
Clone the Repository:

Bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Secrets:
Create a .streamlit/secrets.toml file:

Ini, TOML
GROQ_API_KEY = "your_groq_api_key_here"
Run the App:

Bash
streamlit run app.py
📂 Project Structure
app.py: The core application logic and UI.

requirements.txt: List of necessary Python libraries.

.streamlit/secrets.toml: Secure storage for API keys.

assets/: (Optional) Folder for custom logos/images.

🛡️ Security & Privacy
Secure Secrets: API keys are never hardcoded.

Context Safety: Uses regular expressions to clean non-UTF characters before processing.

Memory Management: Uses @st.cache_resource for efficient LLM loading.

🤝 Contributing
Contributions are welcome! If you have suggestions for new analysis types or UI improvements, feel free to open an issue or submit a pull request.

Developed with ❤️ for the Research Community.
Powered by Groq + Llama 3.3.

