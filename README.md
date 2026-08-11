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
* LangChain
* Streamlit
* Ollama
* FAISS / ChromaDB
* Hugging Face Embeddings
* PyPDF
* RAG (Retrieval-Augmented Generation)

## How It Works

The application follows this process:

```text
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
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project folder:

```bash
cd PDF-Chatbot-RAG
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Ollama Setup

This project can use Ollama to run the language model locally without requiring an API key.

Install Ollama and make sure the required model is available on your system.

For example:

```bash
ollama pull llama3
```

Make sure Ollama is running before starting the application.

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Example Questions

After uploading a PDF, you can ask questions such as:

* What are the projects mentioned in the document?
* What technologies were used in the PDF ChatBot project?
* What technologies were used in the House Price Prediction project?
* What is the educational background?
* What information is available about the projects?

## Project Structure

```text
PDF-Chatbot-RAG/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements

* Support multiple PDF uploads
* Add conversation history
* Improve retrieval accuracy
* Add source/page references for answers
* Add document preview
* Deploy the application online

## Author

Bharath Kumar
