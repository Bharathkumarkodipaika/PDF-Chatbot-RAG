# PDF ChatBot Using RAG

An AI-powered PDF question-answering chatbot that allows users to upload a PDF document and ask questions about its content.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded PDF and generate context-aware answers.

## Features

* Upload PDF documents
* Extract text from PDF files
* Split documents into smaller chunks
* Generate embeddings for document chunks
* Retrieve relevant chunks based on the user's question
* Generate answers using a local LLM
* Display retrieved PDF content
* Interactive web interface using Streamlit
* No API key required when using Ollama locally

## Technologies Used

* Python
* Streamlit
* LangChain
* PyPDF
* all-MiniLM-L6-v2 Embedding Model
* ChromaDB
* Ollama(phi3:latest)
* RAG (Retrieval-Augmented Generation)

## How It Works

The application follows this process:

Upload PDF
    ↓
Extract PDF Text
    ↓
Split Text into Chunks
    ↓
Create Embeddings
    ↓
Store in Vector Database
    ↓
User Asks Question
    ↓
Retrieve Relevant Chunks
    ↓
Send Context + Question to LLM
    ↓
Generate Answer

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Move into the project folder:

cd PDF-Chatbot-RAG


Install the required Python packages:

pip install -r requirements.txt

### Ollama Setup

This project uses Ollama to run the language model locally without requiring an API key.

The application uses:

* **Ollama**
* **Model:** `phi3:latest`
* **Temperature:** `0`

Make sure Ollama is installed and running on your computer.

Pull the required model:

ollama pull phi3:latest

You can verify that the model is installed with:


ollama list

Make sure `phi3:latest` appears in the list before starting the application.

## Run the Application

Start the Streamlit application:

python -m streamlit run app.py

The application will open in your browser.

## Example Questions

After uploading a PDF, you can ask questions such as:

* What are the projects mentioned in the document?
* What technologies were used in the PDF ChatBot project?
* What technologies were used in the House Price Prediction project?
* What is the educational background?
* What information is available about the projects?

## Project Structure

PDF-Chatbot-RAG/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## Future Improvements

* Support multiple PDF uploads
* Add conversation history
* Improve retrieval accuracy
* Add source/page references for answers
* Add document preview
* Deploy the application online

## Author

Bharath Kumar
