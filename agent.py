import fitz  # PyMuPDF
import gradio as gr
import subprocess
import json
import time
import requests

# ------------------------------
# OLLAMA CALL FUNCTION
# ------------------------------
def ollama_query(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()["response"]

# ------------------------------
# PDF TEXT EXTRACTOR
# ------------------------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ------------------------------
# AGENT WORKFLOW
# ------------------------------
def research_agent(user_query, pdf):
    # Step 1: Extract text
    text = extract_text_from_pdf(pdf.name)
    
    # Step 2: Generate Summarization Prompt
    summarize_prompt = f"""
    You are a professional researcher. 
    The user is interested in: {user_query}.
    You have read the following paper:

    {text[:6000]}   # (limit context length for local model)

    Please summarize the key points relevant to the user's query.
    """
    
    # Step 3: Call LLM for summarization
    summary = ollama_query(summarize_prompt)
    
    # Step 4: Generate Final Report
    report_prompt = f"""
    Now based on the summary: {summary}

    Create a nice formatted research report for the user.
    """
    
    final_report = ollama_query(report_prompt)
    
    return final_report

# ------------------------------
# UI USING GRADIO
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Personal Research Agent")
    user_query = gr.Textbox(label="Enter your research question:")
    pdf = gr.File(label="Upload related research PDF")
    output = gr.Textbox(label="Generated Report")
    submit = gr.Button("Run Agent")
    submit.click(research_agent, inputs=[user_query, pdf], outputs=output)

demo.launch(inbrowser=True)
