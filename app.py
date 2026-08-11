import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# Path to the PDF
pdf_path = "sample.pdf"

# Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"PDF loaded successfully! Total pages: {len(documents)}")

# Split the PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Create HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

# Create Chroma Vector Database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("Vector Database created successfully!")

# Create Retriever
retriever = vectordb.as_retriever(
    search_kwargs={"k": 2}
)

print("Retriever created successfully!")

# Load Gemini API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

print("Gemini LLM initialized successfully!")

def ask_pdf(question):
    # Retrieve the most relevant chunks
    docs = retriever.invoke(question)

    # Combine the retrieved text into a single context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create the prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer is not present in the context, simply reply:
"I couldn't find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

    # Send the prompt to Gemini
    response = llm.invoke(prompt)

    return response.content

    question = input("Ask a question about the PDF: ")

answer = ask_pdf(question)

print("\nAnswer:")
print(answer)