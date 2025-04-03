import ipywidgets as widgets
from IPython.display import display, clear_output
from embed_text import get_embedding_components, project_query_embedding
from load_llm import get_llm_pipeline
import time
import threading
import csv
import os
import datetime
import psutil

conversation_history = []

def retrieve_context(query, k=3):
    embedding_model, index, docs = get_embedding_components()
    raw_query_embedding = embedding_model.encode([query])
    query_embedding = project_query_embedding(raw_query_embedding)
    distances, indices = index.search(query_embedding, k)
    return [docs[i] for i in indices[0]]

def generate_answer(question):
    context = "\n".join(retrieve_context(question))
    context = ' '.join(context.split()[:700])
    history = "".join([f"Previous Question: {q}\nPrevious Answer: {a}\n" for q, a in conversation_history[-3:]])
    prompt = f"""You are a helpful research assistant. Use the scientific context and prior conversation to answer thoroughly and clearly.

{history}
Context:
{context}

Question: {question}
Answer:"""
    result = get_llm_pipeline()(prompt, max_new_tokens=500, temperature=0.5, do_sample=True)
    answer = result[0]['generated_text'].split("Answer:")[-1].strip()
    conversation_history.append((question, answer))
    return answer

def launch_chat_ui(_):
    clear_output(wait=True)
    
    # --- Setup logging ---
    # Create "logs" folder if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    # Use current date and time to name the CSV file (e.g., "20250401_153045.csv")
    conversation_start = datetime.datetime.now()
    csv_file_name = conversation_start.strftime("%Y%m%d_%H%M%S") + ".csv"
    csv_file_path = os.path.join(logs_dir, csv_file_name)
    # Write CSV header
    with open(csv_file_path, mode="w", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["User Query", "Response", "Response Time (s)", "Memory Utilization (MB)"])
    
    # --- Setup chat UI ---
    chat_box = widgets.Output(layout={
        'border': '1px solid gray',
        'padding': '10px',
        'height': '400px',
        'overflow_y': 'auto'
    })

    user_input = widgets.Text(placeholder="Ask a question...")
    send_button = widgets.Button(description="Send", button_style="primary")
    status_label = widgets.Label("")

    def start_timer():
        start_time_timer = time.time()
        while not timer_stop[0]:
            elapsed = round(time.time() - start_time_timer, 1)
            status_label.value = f"💬 Thinking... {elapsed}s"
            time.sleep(0.1)
        total = round(time.time() - start_time_timer, 2)
        status_label.value = f"✅ Answered in {total} seconds."

    def on_send(b):
        question = user_input.value.strip()
        if not question:
            return

        user_input.value = ""
        timer_stop[0] = False  # reset flag

        with chat_box:
            display(widgets.HTML(f"<b style='color:blue;'>You:</b> {question}"))

        # Record the start time for response time measurement
        start_time = time.time()

        # Start timer in a separate thread
        timer_thread = threading.Thread(target=start_timer)
        timer_thread.start()

        # Generate answer (this blocks)
        answer = generate_answer(question)

        # Stop the timer and update final status
        timer_stop[0] = True
        timer_thread.join()

        # Record the response time
        response_time = round(time.time() - start_time, 2)
        # Measure memory utilization in MB
        process = psutil.Process(os.getpid())
        memory_usage = round(process.memory_info().rss / (1024 * 1024), 2)

        with chat_box:
            display(widgets.HTML(f"<b style='color:green;'>Assistant:</b> {answer}"))

        # Log the conversation data to CSV
        with open(csv_file_path, mode="a", newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([question, answer, response_time, memory_usage])

    timer_stop = [False]  # Mutable flag shared across threads
    send_button.on_click(on_send)

    display(widgets.VBox([
        widgets.HTML("<h3>🤖 HER-2/neu Assistant</h3>"),
        chat_box,
        status_label,
        widgets.HBox([user_input, send_button])
    ]))

start_chatbot_widget = widgets.Button(description="💬 Start Chatbot", button_style="primary")
output = widgets.Output()
start_chatbot_widget.on_click(launch_chat_ui)
start_chatbot_widget = widgets.VBox([start_chatbot_widget, output])
