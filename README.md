# PDF Q&A with AstraDB, LangChain, and Gradio

This project is a simple **PDF Question-Answering application** built
using:

-   **LangChain** for LLM and vector store
-   **Groq LLM (openai/gpt-oss-120b)** as the chatbot model
-   **HuggingFace Sentence Transformer embeddings** for vectorization/embedding
-   **Astra DB (Cassandra)** as the vector database
-   **Gradio** for a clean UI

The app allows you to: 1. Upload a PDF/Document 2. Extract and index its text
into Astra DB 3. Ask questions and get answers base on the PDF
content


## Features

-   PDF text extraction using **PyPDF2**
-   Chunking and embedding with **CharacterTextSplitter** + **HuggingFaceEmbeddings** respectively
-   Vector search powered by **Astra DB**
-   Real-time LLM responses using **Groq**
-   Interactive UI built with **Gradio**


## Project Structure

    Final_pdf_query_LangChain_30082025.py   # Main application file
    README.md       # Documentation
    requirements.txt    # Required packages


## Installation

Make sure you have Python 3.9+ installed.

Install required packages:

``` bash
pip install gradio langchain langchain-community langchain-groq     langchain-google-genai langchain-vectorstores cassio PyPDF2     sentence-transformers
```

## Set Up API Keys

The script requires the following: - **Groq API key** - **ASTRA_DB_APPLICATION_TOKEN + ASTRA_DB_ID**

Set them as environment variables:

``` bash
export GROQ_API_KEY="your_key"
export ASTRA_DB_APPLICATION_TOKEN="your_token"
export ASTRA_DB_ID="your_db_id"
```


## Running the App

Run the script:

``` bash
python Final_pdf_query_LangChain_30082025.py
```

This will launch a **Gradio interface** in your browser.


## How It Works

### 1. Upload PDF

The app extracts text using `PyPDF2.PdfReader`.

### 2. Chunk + Embed

Text is split into 800-character chunks with 200 overlap. Embeddings are
generated using **MiniLM-L6-v2**.

### 3. Store in Astra DB

Chunks and embeddings are stored using LangChain's Cassandra
integration.

### 4. Ask Questions

Queries go through: - Vector similarity search - Groq LLM for final
answer

## Important Notes

-   Gemini embeddings free-tier is unavailable; MiniLM is used instead.
-   Groq model `openai/gpt-oss-120b` powers the chat responses.
-   Vector index table name: **pdf_qa_query_db_new**


## Contributing

Feel free to create issues or pull requests.


## License

This project is for educational and personal use only.
