# Gradio-Rag-Chatbot
RAG chatbot integrating retrieval and LLMs

# Financial Report Analysis Chatbot  

This project involves creating a Retrieval-Augmented Generation (RAG) chatbot capable of answering queries related to financial reports, current events, and general questions. The chatbot processes financial documents, integrates retrieval and query routing workflows, and delivers interactive responses via a Gradio application deployed on HuggingFace Spaces.  

## Features  

### 1. Process the PDF  
- Used **unstructured.io** to ingest and extract structured data from financial reports.  
- Ensured proper handling of complex tables and detailed data points.  

### 2. Develop Query Routing  
- Implemented a system to classify user queries into the following categories:  
  - **Finance-related:** Queries concerning the financial report.  
  - **Live-event related:** Queries about current events.  
  - **General:** Other inquiries.  

### 3. Build RAG Workflow  
- Integrated a retrieval mechanism using an embedding model and a vector database to address finance-related questions.  
- Used **Tavily** for internet search to handle current event queries.  
- Connected all modules with a **large language model (LLM)** to generate accurate responses.  

### 4. Implement Evaluation Metrics  
- Used **RAGAS metrics** to evaluate:  
  - **Correctness:** Ensures the response addresses the query appropriately.  
  - **Faithfulness:** Verifies that the response is grounded in retrieved data.  
- **Evaluation Results:**  
  - **Total Questions:** 22  
  - **Correctness:** 100%  
  - **Faithfulness:** 58.59%  

### 5. Create a Gradio App  
- Built an interactive user interface using **Gradio**.  
- Deployed the app on **HuggingFace Spaces**, making it publicly accessible for testing.  

## Installation  

To set up the project locally, follow these steps:  

1. Clone the repository:  
   ```bash
   git clone <"https://github.com/Ayesha-Zafar1/Gradio-Rag-Chatbot/">
   cd <"Gradio-Rag-Chatbot">
   
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

# Requirements
The project requires the following Python libraries:

- torch
- langchain
- sentence_transformers
- gradio
- pinecone-client
- tavily-python
- pytesseract
- pdfminer
- pdf2image
- IPython
- pdfminer.six
- requests
- groq

# Usage
1. Run the Gradio application locally:
   ```bash
   python app.py

3. Access the app through the provided URL or visit the deployed version on HuggingFace Spaces.
   URL: "https://huggingface.co/spaces/Ayesha-15/gradio-rag-chatbot"

# Project Architecture
- PDF Processing: Converts unstructured financial data into structured format for querying.
- Query Routing: Routes queries to appropriate modules based on category.
- RAG Workflow: Uses embedding-based retrieval and internet search to generate accurate responses.
- Evaluation: Measures response quality using RAGAS metrics.
- Gradio Interface: Enables user interaction with the chatbot.

# Deployment
The application is deployed on HuggingFace Spaces.

# Results
- Correctness: 100%
- Faithfulness: 58.59%
  
# Future Improvements
- Enhance faithfulness of responses by refining retrieval mechanisms.
  Expand query classification to include additional categories.
- Improve user experience with advanced interactive features.

# License
This project is licensed under the MIT License.

# Acknowledgments
- Unstructured.io for PDF processing.
- Tavily for live event search integration.
- HuggingFace Spaces for deployment.
- Gradio for building an interactive user interface.


