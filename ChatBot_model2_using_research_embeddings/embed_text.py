from sentence_transformers import SentenceTransformer
import faiss
import ipywidgets as widgets
from IPython.display import display
import pickle
import numpy as np
import pandas as pd

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
    spinner.value = "📌 Loading precomputed clinical embeddings... Please wait."

    try:
        # Load precomputed clinical embeddings (assumed to be 128-dimensional)
        embed_mat = pickle.load(open("full_h_embed_hms.pkl", "rb")).numpy()
        
        # Load the mapping file to get clinical concept descriptions
        node_map_df = pd.read_csv("new_node_map_df.csv")
        node_map_df.sort_values("global_graph_index", inplace=True)
        docs = node_map_df["node_name"].tolist()
        
        # For query encoding, we use a model that outputs higher-dimensional vectors.
        # Here, we use "all-mpnet-base-v2" as an example.
        embedding_model = SentenceTransformer("all-mpnet-base-v2")
        
        dim = embed_mat.shape[1]  # should be 128
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 5)
        index.train(embed_mat)
        index.add(embed_mat)
        index.nprobe = 3

        spinner.value = ""
        with output:
            print("✅ Precomputed clinical embeddings loaded successfully.")
            print("➡️ Now click '🚀 Load LLM'")
    except Exception as e:
        spinner.value = ""
        with output:
            print("❌ Error loading precomputed embeddings:", str(e))

create_embedding_button = widgets.Button(description="📌 Load Precomputed Embeddings", button_style="warning")
create_embedding_button.on_click(handle_embedding)

create_embedding_widget = widgets.VBox([create_embedding_button, spinner, output])

# Projection for query embeddings (placeholder; ideally learn this projection)
query_proj = None
def project_query_embedding(query_embedding):
    global query_proj
    if query_proj is None:
        D = query_embedding.shape[1]  # e.g., 768 from all-mpnet-base-v2
        query_proj = np.random.randn(D, 128).astype("float32")
    return np.dot(query_embedding, query_proj)

def get_embedding_components():
    return embedding_model, index, docs
