# Interactive ChatBot (RAG-based Chatbot)

An interactive chatbot for scientific literature exploration powered by Retrieval-Augmented Generation (RAG). This tool allows researchers and students to upload a HER-2/neu-related PDF, generate embeddings for document understanding, and ask contextual questions using an integrated LLM pipeline. I **developed two chatbots**: one using Clinical-knowledge-embeddings(full_h_embed_hms.pkl) and 2nd one Lightweight sentence transformer for encoding both document chunks and queries(intfloat/e5-small-v2).

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

- Upload all required `.py` files (e.g., `main_interface.py`, `upload_pdf.py`, etc.) from the **ChatBotUsingRAG** folder to the Colab environment.
- Open **`ChatBotUsingRAGUpload.ipynb`** in [Google Colab](https://colab.research.google.com/).
- Upload your **PDF file**.
- Click **🔗 Create Embeddings**.
- Click **⚙️ Load LLM** (Falcon 7B).
- Click **💬 Start Chatbot** and begin asking questions!


> 📌 All instructions are included within the notebook.

---

## Resources on Google Colab

| Requirement         | Recommended |
|---------------------|-------------|
| 💾 System RAM        | Minimum **35 GB** |
| 💽 Disk Space        | Minimum **40 GB** |
| 🚀 Hardware Accelerator | **v2-8 TPU** or better |

> ✅ **Ensure you have GPU enabled in your Colab Runtime for optimal performance.**

To enable GPU in Colab:
- Click `Runtime` → `Change runtime type` → Set **Hardware accelerator** to `GPU` or `TPU`.


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






