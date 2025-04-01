# from upload_pdf import upload_file_widget, get_full_text
# from embed_text import create_embedding_widget, set_full_text
# from load_llm import load_llm_widget
# from chatbot_core import start_chatbot_widget

# import ipywidgets as widgets
# from IPython.display import display

# # Connect the full_text from upload -> embed_text
# set_full_text(get_full_text)

# # Final UI
# display(widgets.VBox([
#     widgets.HTML("<h2>📚 HER-2/neu Interactive Assistant</h2>"),
#     upload_file_widget,
#     create_embedding_widget,
#     load_llm_widget,
#     start_chatbot_widget
# ]))

from upload_pdf import upload_file_widget, get_full_text
from embed_text import create_embedding_widget, set_full_text
from load_llm import load_llm_widget
from chatbot_core import launch_chat_ui

import ipywidgets as widgets
from IPython.display import display, clear_output

# Connect full_text -> embed_text
set_full_text(get_full_text)

# === PHASE 1: SETUP UI ===
setup_complete = [False]  # use mutable object to preserve across callbacks
status_label = widgets.Label()

def finish_setup(b):
    setup_complete[0] = True
    status_label.value = "✅ Setup complete. You can now start chatting anytime below."
    run_chat_button.layout.display = "inline-block"

# Button to confirm setup is done
complete_button = widgets.Button(description="✅ Setup Complete", button_style="success")
complete_button.on_click(finish_setup)

setup_ui = widgets.VBox([
    widgets.HTML("<h2>📚 HER-2/neu Research Assistant: Setup</h2>"),
    upload_file_widget,
    create_embedding_widget,
    load_llm_widget,
    complete_button,
    status_label
])

# === PHASE 2: CHATBOT LAUNCH ===
def launch_chatbot(b):
    clear_output(wait=True)
    launch_chat_ui(None)

run_chat_button = widgets.Button(description="💬 Launch Chatbot", button_style="primary")
run_chat_button.on_click(launch_chatbot)
run_chat_button.layout.display = "none"  # hidden until setup is confirmed

chat_launcher_ui = widgets.VBox([
    widgets.HTML("<h2>🤖 Start Chatting</h2>"),
    run_chat_button
])

# === DISPLAY FULL INTERFACE ===
display(widgets.VBox([
    setup_ui,
    chat_launcher_ui
]))
