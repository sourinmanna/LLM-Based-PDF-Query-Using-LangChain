import gradio as gr
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import cassio
import os

# =============================
# Gemini API Key Setup
# =============================
GEMINI_API_KEY = "AIzaSyBeCIdxI1tJGgEoZWSWPNoUhxVg1KsBqaw"  # replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# =============================
# Astra DB Key Setup
# =============================
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:oFZznHjGyBYtvzOuSwGQIzpe:e25e4eebd1d3d9ea369060a9e4bba93b069f9a019b60f91b59164ae029a8b305"
ASTRA_DB_ID = "9497d5e9-4443-4034-9f00-9f6e81cfa05b"

# =============================
# Init Astra DB
# =============================
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# =============================
# LLM + Embeddings (Gemini)
# =============================
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# =============================
# Vector Store (Astra DB)
# =============================
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="pdf_qa_query_db_new",
    session=None,
    keyspace=None,
)

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# =============================
# PDF Processing + Indexing
# =============================
def process_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file first."

    pdfreader = PdfReader(pdf_file.name)
    raw_text = ""
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    astra_vector_store.add_texts(texts)
    return "PDF processed and indexed successfully!"

# =============================
# QA Function
# =============================
def answer_question(question):
    if not question.strip():
        return "Please enter a question."
    answer = astra_vector_index.query(question, llm=llm).strip()
    return answer

# =============================
# Gradio UI
# =============================
with gr.Blocks() as demo:
    gr.Markdown("# 📘 PDF Q&A with AstraDB + LangChain + Gemini API")

    with gr.Row():
        # Left side (PDF Upload)
        with gr.Column(scale=1):
            pdf_upload = gr.File(label="Upload your PDF", file_types=[".pdf"])
        
        # Right side (Process button on top, status on bottom)
        with gr.Column(scale=1):
            process_btn = gr.Button("Process PDF")
            status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer", interactive=False)

    process_btn.click(fn=process_pdf, inputs=pdf_upload, outputs=status_output)
    question_input.submit(fn=answer_question, inputs=question_input, outputs=answer_output)

# Run app
demo.launch()
