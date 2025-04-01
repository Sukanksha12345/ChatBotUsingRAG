import ipywidgets as widgets
from IPython.display import display, clear_output
from embed_text import get_embedding_components, project_query_embedding
from load_llm import get_llm_pipeline
import time
import threading

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
        start_time = time.time()
        while not timer_stop[0]:
            elapsed = round(time.time() - start_time, 1)
            status_label.value = f"💬 Thinking... {elapsed}s"
            time.sleep(0.1)
        total = round(time.time() - start_time, 2)
        status_label.value = f"✅ Answered in {total} seconds."

    def on_send(b):
        question = user_input.value.strip()
        if not question:
            return

        user_input.value = ""
        timer_stop[0] = False  # reset flag

        with chat_box:
            display(widgets.HTML(f"<b style='color:blue;'>You:</b> {question}"))

        # Start timer in a separate thread
        timer_thread = threading.Thread(target=start_timer)
        timer_thread.start()

        # Generate answer (this blocks)
        answer = generate_answer(question)

        # Stop the timer and update final status
        timer_stop[0] = True
        timer_thread.join()

        with chat_box:
            display(widgets.HTML(f"<b style='color:green;'>Assistant:</b> {answer}"))

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
