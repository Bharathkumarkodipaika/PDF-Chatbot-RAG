import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Chatbot - RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF Chatbot using RAG")

st.write(
    "Upload a PDF and ask questions about its content "
    "using Retrieval-Augmented Generation (RAG)."
)

st.divider()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    st.success(
        f"PDF uploaded: {uploaded_file.name}"
    )


    # --------------------------------------------------
    # Load PDF
    # --------------------------------------------------

    with st.spinner("Loading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    st.info(
        f"PDF loaded successfully! Total pages: {len(documents)}"
    )


    # --------------------------------------------------
    # Split PDF into chunks
    # --------------------------------------------------

    with st.spinner("Splitting PDF into chunks..."):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = text_splitter.split_documents(documents)

    st.info(
        f"Total chunks created: {len(chunks)}"
    )


    # --------------------------------------------------
    # Create HuggingFace Embeddings
    # --------------------------------------------------

    with st.spinner("Loading embedding model..."):

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    st.success(
        "Embedding model loaded successfully!"
    )


    # --------------------------------------------------
    # Create Chroma Vector Database
    # --------------------------------------------------

    with st.spinner("Creating vector database..."):

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

    st.success(
        "Chroma vector database created successfully!"
    )


    # --------------------------------------------------
    # Create Retriever using MMR
    # --------------------------------------------------

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 7
        }
    )


    # --------------------------------------------------
    # Initialize Ollama
    # --------------------------------------------------

    try:

        llm = ChatOllama(
            model="phi3:latest",
            temperature=0
        )

    except Exception as e:

        st.error(
            "Could not initialize Ollama. "
            "Make sure Ollama is running and phi3:latest is installed."
        )

        st.exception(e)

        st.stop()

#```python
# --------------------------------------------------
# Ask Question
# --------------------------------------------------

question = st.text_input(
    "Ask a question about your PDF:"
)

if question:

    with st.spinner(
        "Searching the PDF and generating answer..."
    ):

        # Retrieve relevant chunks
        docs = retriever.invoke(question)

        # Combine retrieved content
        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Create prompt
        prompt = f"""
You are a document question-answering assistant.

Your task is to answer the user's question using ONLY the
information explicitly present in the provided PDF context.

STRICT RULES:

1. Do not add information that is not present in the context.
2. Do not make assumptions or guesses.
3. Do not infer dates, colleges, tools, or experiences unless
   they are explicitly stated in the context.
4. If the question asks about PROJECTS, include only items
   explicitly listed under the PROJECTS section.
5. Do not treat certifications, achievements, skills, or
   education as projects.
6. If the answer cannot be found in the context, respond exactly:
   "I couldn't find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        response = llm.invoke(prompt)

        # Display Answer
        st.subheader("Answer")

        st.write(response.content)

        # Display Retrieved Chunks
        with st.expander(
            "View Retrieved PDF Content"
        ):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"**Retrieved Chunk {i + 1}**"
                )

                st.write(
                    doc.page_content
                )

                st.divider()

