from google.colab import files
import pdfplumber
import ipywidgets as widgets
from IPython.display import display

# Store PDF text globally
full_text = ""

# This is what the notebook imports
def get_full_text():
    return full_text

# Spinner/Status widget
spinner = widgets.Label(value="")

# Output for logs
output = widgets.Output()

def handle_upload(b):
    global full_text
    output.clear_output()
    spinner.value = "⏳ Uploading and processing PDF... Please wait."

    try:
        uploaded = files.upload()
        filename = next(iter(uploaded))
        with pdfplumber.open(filename) as pdf:
            full_text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        
        spinner.value = ""
        with output:
            if full_text.strip():
                print(f"✅ File '{filename}' uploaded and processed successfully.")
                print("➡️ Now click '📌 Create Embeddings'")
            else:
                print("❌ No valid text found in PDF. Try another file.")
    except Exception as e:
        spinner.value = ""
        with output:
            print("❌ Error during upload or processing:", str(e))

upload_file_button = widgets.Button(description="📁 Upload PDF", button_style="info")
upload_file_button.on_click(handle_upload)

upload_file_widget = widgets.VBox([upload_file_button, spinner, output])
