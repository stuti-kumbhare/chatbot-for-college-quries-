import json
import random
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
with open("intents.json", "r") as file:
    data = json.load(file)

# Prepare training data
patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern.lower())
        tags.append(intent["tag"])

# Train NLP model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

# Chat function
def get_response(user_input):
    user_input = user_input.lower()

    if user_input == "bye":
        return "Goodbye! Have a great day."

    user_vector = vectorizer.transform([user_input])
    predicted_tag = model.predict(user_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand that question."

# Send button function
def send_message():
    user_message = entry_box.get()

    if user_message.strip() == "":
        return

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"You: {user_message}\n")

    bot_response = get_response(user_message)
    chat_area.insert(tk.END, f"Bot: {bot_response}\n\n")

    chat_area.config(state=tk.DISABLED)
    chat_area.see(tk.END)

    entry_box.delete(0, tk.END)

# GUI Window
root = tk.Tk()
root.title("BIST College Queries Chatbot")
root.geometry("700x550")

title_label = tk.Label(
    root,
    text="BIST College Queries Chatbot",
    font=("Arial", 16, "bold")
)
title_label.pack(pady=10)

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=80,
    height=25
)
chat_area.pack(padx=10, pady=10)
chat_area.config(state=tk.DISABLED)

entry_box = tk.Entry(root, width=60, font=("Arial", 12))
entry_box.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

send_button = tk.Button(
    root,
    text="Send",
    command=send_message,
    width=12
)
send_button.pack(side=tk.RIGHT, padx=10, pady=10)

root.mainloop()