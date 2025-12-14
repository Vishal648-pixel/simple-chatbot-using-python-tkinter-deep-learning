import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
import os
import sys

# Ensure NLTK resources (safe to call repeatedly)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------- Model and data loading (replaced to prefer chatbot_model.h5) ----------------
MODEL_FILES = ["chatbot_model.h5", "model.h5", "chatbot.h5"]
model = None
for mf in MODEL_FILES:
    if os.path.exists(mf):
        try:
            model = load_model(mf)
            print(f"Loaded model: {mf}")
            break
        except Exception as e:
            print(f"Found {mf} but failed to load: {e}", file=sys.stderr)

if model is None:
    raise FileNotFoundError(
        "No model file found. Run train_chatbot.py to create 'chatbot_model.h5' or place a valid model file in the same folder."
    )

try:
    intents = json.loads(open('intents.json', encoding='utf-8').read())
except Exception as e:
    raise FileNotFoundError(f"Could not open 'intents.json': {e}")

try:
    words = pickle.load(open('words.pkl', 'rb'))
except Exception as e:
    raise FileNotFoundError(f"Could not open 'words.pkl': {e}")

try:
    classes = pickle.load(open('classes.pkl', 'rb'))
except Exception as e:
    raise FileNotFoundError(f"Could not open 'classes.pkl': {e}")

# ---------------- Original functions (kept intact) ----------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=np.float32)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "noanswer"

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return

    # Insert user message
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_input + "\n")
    chat_area.config(state=tk.DISABLED)

    tag = predict_class(user_input)
    response = get_response(tag)

    # Insert bot response
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "Bot: " + response + "\n\n")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END)
    entry.delete(0, tk.END)

# ---------------- GUI (only GUI replaced) ----------------

# Root window
root = tk.Tk()
root.title("Healthcare Chatbot")
root.geometry("520x620")
root.resizable(False, False)
root.configure(bg="#f5f7fb")

# Top header
header = tk.Frame(root, bg="#2b7de9", height=70)
header.pack(fill="x")
header.pack_propagate(False)
title = tk.Label(header, text="HEALTHCARE CHATBOT", bg="#2b7de9", fg="white",
                 font=("Helvetica", 14, "bold"))
title.pack(side="left", padx=16)
status = tk.Label(header, text="Online", bg="#2b7de9", fg="#e6f7ff", font=("Helvetica", 10))
status.pack(side="right", padx=12)

# Main chat frame with panel effect
panel = tk.Frame(root, bg="#ffffff", bd=0, relief="flat")
panel.place(x=10, y=80, width=500, height=420)

# Chat area (scrolledtext) inside panel
chat_area = scrolledtext.ScrolledText(panel, wrap=tk.WORD, font=("Arial", 11), bg="#f8fbff",
                                      bd=0, padx=10, pady=10)
chat_area.pack(fill="both", expand=True)
chat_area.insert(tk.END, "Bot: Hello! How can I help you today?\n\n")
chat_area.config(state=tk.DISABLED)

# Bottom input frame
bottom_frame = tk.Frame(root, bg="#f5f7fb", height=110)
bottom_frame.pack(side="bottom", fill="x")
bottom_frame.pack_propagate(False)

# Entry box with placeholder behavior
entry_var = tk.StringVar()
entry = tk.Entry(bottom_frame, textvariable=entry_var, font=("Arial", 12), bd=1, relief="solid")
entry.place(x=16, y=20, width=380, height=44)
entry.focus_set()

# Send button (styled)
send_btn = tk.Button(bottom_frame, text="Send", font=("Arial", 11, "bold"),
                     bg="#2b7de9", fg="white", bd=0, activebackground="#1f6fd6",
                     command=send_message)
send_btn.place(x=410, y=20, width=90, height=44)

# Quick action buttons (optional)
def quick_text(t):
    entry_var.set(t)
    entry.icursor(tk.END)

quick_frame = tk.Frame(bottom_frame, bg="#f5f7fb")
quick_frame.place(x=16, y=70, width=484, height=30)
q1 = tk.Button(quick_frame, text="Hi", command=lambda: quick_text("hi"), bd=0, bg="#eef6ff")
q1.pack(side="left", padx=6)
q2 = tk.Button(quick_frame, text="Good morning", command=lambda: quick_text("good morning"), bd=0, bg="#eef6ff")
q2.pack(side="left", padx=6)
q3 = tk.Button(quick_frame, text="Thank you", command=lambda: quick_text("thank you"), bd=0, bg="#eef6ff")
q3.pack(side="left", padx=6)

# Bind Enter key to send (keeps original send_message function)
root.bind('<Return>', lambda event: send_message())

# Keep window centered on screen
root.update_idletasks()
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
size = tuple(int(_) for _ in root.geometry().split('+')[0].split('x'))
x = w//2 - size[0]//2
y = h//2 - size[1]//2
root.geometry(f"{size[0]}x{size[1]}+{x}+{y}")

root.mainloop()
