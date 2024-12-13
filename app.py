#from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from google.colab import files
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
#from gradio import gr
import gradio as gr
from IPython.display import display
import requests
import os
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract
from langchain.prompts import PromptTemplate
import json

# Upload PDF
# uploaded = files.upload()
# pdf_path = next(iter(uploaded.keys()))

pdf_path = "./NetSol_Financial Statement_2024_Part 1.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        if text.strip():
            return text
    except Exception as e:
        print("Failed to extract text using pdfminer:", e)

    # Fallback to OCR if pdfminer fails
    print("Using Tesseract for OCR on scanned PDF...")
    images = convert_from_path(pdf_path)
    text = ""
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()[]{}^~<>πσ√'

    for i, image in enumerate(images):
        text += pytesseract.image_to_string(image, config=custom_config) + "\n"

    return text

# # Upload PDF
# uploaded = files.upload()  # Prompt for file upload
# pdf_path = next(iter(uploaded.keys()))

# Extract text from the uploaded PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Split the extracted text into chunks using LangChain
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.create_documents([extracted_text])

# Load Sentence-BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embeddings(texts):
    return model.encode([text.page_content for text in texts])

# Generate embeddings for the chunks
embeddings = generate_embeddings(chunks)

# Print the embeddings
# print("Generated Embeddings:")
# for i, embedding in enumerate(embeddings):
#     print(f"Chunk {i + 1}: {embedding[:10]}...")  # Print first 10 values of each embedding for brevity

from pinecone import Pinecone, ServerlessSpec, Index
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from google.colab import files

# Initialize Pinecone client
pc = Pinecone(api_key="pcsk_8drpB_AMNSB8Tnur3jrKkt9egkSfezH8xGuDcToYaCQ5e9qarYAmXqor2w6uY5ph9qV1s")  # Use your actual API key

# List indexes
indexes = pc.list_indexes().names()
print("Indexes:", indexes)

# If an index exists, describe it, else create a new index
if indexes:
    index_name = indexes[0]  # Access the first index in the list and get the 'name'
    index_description = pc.describe_index(index_name)  # Pass the index name as a string
    print(f"Details of index '{index_name}':", index_description)
else:
    # Create a new index if none exist
    pc.create_index(
        name='netsol_financials',
        dimension=384,  # Update dimension according to your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Set your region here
        )
    )
    print("Created 'netsol_financials' index.")

# Access the 'notes' index using the Index class
index = Index(name='netsol_financials', host='https://netsol-financials-c2umrz9.svc.aped-4627-b74a.pinecone.io', api_key="pcsk_8drpB_AMNSB8Tnur3jrKkt9egkSfezH8xGuDcToYaCQ5e9qarYAmXqor2w6uY5ph9qV1s")

# Prepare data for insertion into Pinecone
vectors_to_insert = []
for i, embedding in enumerate(embeddings):
    vectors_to_insert.append((
        str(i),  # Unique ID for each vector
        embedding.tolist(),  # Embedding vector
        {"chunk_id": i, "content": chunks[i].page_content[:200]}  # Optional metadata
    ))

# Insert the embeddings into Pinecone
index.upsert(vectors=vectors_to_insert)

# Verify insertion by querying the Pinecone index (optional)
query_result = index.query(vector=embeddings[0].tolist(), top_k=3)  # Corrected to use keyword arguments
#print("Query Result:", query_result)

# Optionally, check the status of the index
status = index.describe_index_stats()
#print("Index Stats:", status)

import os
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index
from tavily import TavilyClient

# Set the API key for Groq
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key="pcsk_8drpB_AMNSB8Tnur3jrKkt9egkSfezH8xGuDcToYaCQ5e9qarYAmXqor2w6uY5ph9qV1s")  # Use your actual API key

# List indexes
indexes = pc.list_indexes().names()
# If an index exists, describe it, else create a new index
if indexes:
    index_name = indexes[0]  # Access the first index in the list and get the 'name'
    index_description = pc.describe_index(index_name)  # Pass the index name as a string
else:
    # Create a new index if none exist
    pc.create_index(
        name='notes',
        dimension=384,  # Update dimension according to your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Set your region here
        )
    )

# Access the 'notes' index using the Index class
index = Index(name='netsol_financials', host='https://netsol-financials-c2umrz9.svc.aped-4627-b74a.pinecone.io', api_key="pcsk_8drpB_AMNSB8Tnur3jrKkt9egkSfezH8xGuDcToYaCQ5e9qarYAmXqor2w6uY5ph9qV1s")

# Load Sentence-BERT model for generating query embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate embeddings for user input
def generate_embeddings(texts):
    return model.encode(texts)


# Function to classify queries
def classify_query(query):
    if any(keyword in query.lower() for keyword in ["finance", "financial", "stock", "investment", "market", "NetSol"]):
        return "finance"
    elif any(keyword in query.lower() for keyword in ["news", "event", "current", "recent", "happening now", "live"]):
        return "live-event"
    else:
        return "general"

# # Function to generate embeddings for user input
# def generate_embeddings(texts):
#     return model.encode(texts)

def get_tavily_response(query: str) -> str:
    """
    Fetch current event data using Tavily API via TavilyClient.
    Parses the response to extract relevant titles and URLs.
    """
    from tavily import TavilyClient

    # Initialize the Tavily client
    tavily_client = TavilyClient(api_key="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")

    try:
        # Perform the search query
        response = tavily_client.search(query)

        # Check for results and parse appropriately
        if "results" in response and len(response["results"]) > 0:
            # Extract and format the top results
            formatted_results = "\n\n".join(
                f"**Title**: {item['title']}\n**URL**: {item['url']}\n**Content Snippet**: {item['content'][:200]}..."
                for item in response["results"][:3]
            )
            return f"Top Results:\n{formatted_results}"
        else:
            return "No relevant results found for the query."
    except Exception as e:
        # Handle errors gracefully
        return f"Error fetching data from Tavily: {str(e)}"

# Function to get a response from Groq (LLaMA model)
def get_response_from_lama(prompt: str, model: str = "llama3-8b-8192") -> str:
    """
    Fetch a response from the LLaMA model using the Groq client library.
    Assumes the Groq client is initialized and accessible as `client`.
    """
    try:
        # Call the Groq API using the client
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7
        )
        # Return the content of the response
        return chat_completion.choices[0].message.content
    except Exception as e:
        # Raise an exception with details on failure
        raise ValueError(f"Groq API call failed: {e}")

def get_answer_from_query(user_query: str):
    # Classify the query
    category = classify_query(user_query)
    query_embedding = generate_embeddings([user_query])[0]

    if category == "finance":
        # Query Pinecone with the embedding and get metadata if available
        results = index.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True)

        # Extract the context from metadata if available
        if 'matches' in results:
            context = "\n".join([res['metadata']['content'] for res in results['matches'] if 'metadata' in res])
        else:
            context = "No context found in metadata."
    elif category == "live-event":
        context = get_tavily_response(user_query)
    else:
        return get_response_from_lama(f"User Query: {user_query}\nAnswer:")

    prompt = f"Context:\n{context}\nUser Query: {user_query}\nAnswer:"
    return get_response_from_lama(prompt)


# User input query
# user_query = input("Please enter your query: ")

# # Get the answer from the Pinecone + LLaMA pipeline
# answer = get_answer_from_query(user_query)

# # Print only the question and the answer
# print(f"Question: {user_query}")
# print(f"Answer: {answer}")
#NetSol


#Function to create a Gradio app
def create_gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("## Retrieval-Augmented Generation Chatbot")
        query = gr.Textbox(label="Enter your query")
        output = gr.Textbox(label="Answer")

        def on_query(query):
            return get_answer_from_query(query)

        query.submit(on_query, inputs=query, outputs=output)

    return app

# Create Gradio app and launch it
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
    
