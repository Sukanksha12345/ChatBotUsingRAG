from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ipywidgets as widgets
from IPython.display import display
import torch

llm_pipeline = None

# Spinner/Status widget
spinner = widgets.Label(value="")
output = widgets.Output()

def handle_llm(b):
    global llm_pipeline
    output.clear_output()
    spinner.value = "🚀 Loading LLM... This may take 30–60 seconds. Please wait."

    try:
        model_name = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

        spinner.value = ""
        with output:
            print("✅ LLM pipeline loaded successfully.")
            print("➡️ Now click '💬 Start Chatbot' to begin chatting.")
    except Exception as e:
        spinner.value = ""
        with output:
            print("❌ Error loading the LLM:", str(e))

load_llm_button = widgets.Button(description="🚀 Load LLM", button_style="success")
load_llm_button.on_click(handle_llm)

load_llm_widget = widgets.VBox([load_llm_button, spinner, output])

def get_llm_pipeline():
    return llm_pipeline
