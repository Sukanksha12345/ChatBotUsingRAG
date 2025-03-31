# HER-2/neu Research Assistant (RAG-based Chatbot)

An interactive chatbot for scientific literature exploration powered by Retrieval-Augmented Generation (RAG). This tool allows researchers and students to upload a HER-2/neu-related PDF, generate embeddings for document understanding, and ask contextual questions using an integrated LLM pipeline.

## Project Structure

| File/Directory        | Description                                           |
|----------------------|-------------------------------------------------------|
| `ChatBotUsingRAG.ipynb` | Notebook interface for launching the assistant      |
| `main_interface.py`     | Modular UI for production-style interaction         |
| `upload_pdf.py`         | Handles PDF upload and text extraction              |
| `embed_text.py`         | Chunks document and creates FAISS embeddings        |
| `load_llm.py`           | Loads LLM (e.g., HuggingFace pipeline)              |
| `chatbot_core.py`       | Core logic for retrieval, prompting, chat UI        |

## Features

- Upload and extract text from PDF research papers
- Generate semantic embeddings using SentenceTransformers + FAISS
- Efficient similarity-based context retrieval
- LLM-driven conversational interface for asking research questions
- Runs in Jupyter/Colab via interactive widgets

## How to Use (in Google Colab)

1. Open **ChatBotUsingRAGUpload.ipynb** on goole colab and it has all the instructions.
2. Upload the PDF File
3. Click **Create Embeddings**.
4. Click **Load LLM** (Falcon 7B).
5. Click **Start Chatbot** and begin asking questions!

## Model Used

- Embedding Model: `intfloat/e5-small-v2`
- LLM: `tiiuae/falcon-7b-instruct`

## Technologies Used

- pdfplumber: PDF text extraction
- sentence-transformers: Embedding generation
- faiss-cpu: Approximate nearest neighbor search
- transformers: LLM pipeline integration
- ipywidgets: Rich interactive UI components
- Google Colab or Jupyter: Execution environment

## Contact
For questions, suggestions, or collaborations, please reach out to the Sukanksha Totade @ sukankshatotade1@gmail.com


## Resources on Google Colab
- System RAM : Min 100GB
- Disk - Min 100 GB
- Hardware Accelerator - v2-8 TPU

> Ensure you have GPU enabled in your Colab Runtime for best performance.



