def generate_prompts(text_chunk, mode="summary"):
    if mode == "summary":
        return f"""
        Analyze the following research paper text and provide a concise 
        section-wise summary. Focus on key findings and methodology:
        
        TEXT: {text_chunk}
        """
    elif mode == "rewrite":
        return f"""
        Rewrite the following text to ensure it is 100% plagiarism-free 
        while maintaining the original scientific meaning. Use a professional 
        academic tone but different vocabulary:
        
        TEXT: {text_chunk}
        """

# Example integration with Groq/Gemini
# response = llm.invoke(generate_prompts(chunk, mode="rewrite"))
