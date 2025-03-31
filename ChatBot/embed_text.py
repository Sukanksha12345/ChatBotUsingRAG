from sentence_transformers import SentenceTransformer
import faiss
import ipywidgets as widgets
from IPython.display import display

docs, embedding_model, index = [], None, None
get_text_func = None

def set_full_text(func):
    global get_text_func
    get_text_func = func

def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Spinner/Status widget
spinner = widgets.Label(value="")
output = widgets.Output()

def handle_embedding(b):
    global docs, embedding_model, index
    output.clear_output()
    spinner.value = "📌 Creating embeddings... Please wait."

    try:
        text = get_text_func()
        if not text or not text.strip():
            spinner.value = ""
            with output:
                print("❌ No valid text found. Upload a PDF first.")
            return

        docs = chunk_text(text)
        embedding_model = SentenceTransformer("intfloat/e5-small-v2")
        doc_embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=32)

        dim = doc_embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 5)
        index.train(doc_embeddings)
        index.add(doc_embeddings)
        index.nprobe = 3

        spinner.value = ""
        with output:
            print("✅ Embeddings created successfully.")
            print("➡️ Now click '🚀 Load LLM'")
    except Exception as e:
        spinner.value = ""
        with output:
            print("❌ Error during embedding creation:", str(e))

create_embedding_button = widgets.Button(description="📌 Create Embeddings", button_style="warning")
create_embedding_button.on_click(handle_embedding)

create_embedding_widget = widgets.VBox([create_embedding_button, spinner, output])

def get_embedding_components():
    return embedding_model, index, docs
