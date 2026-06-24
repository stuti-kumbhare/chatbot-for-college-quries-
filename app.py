import streamlit as st
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Chatbot for College Queries",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp{
    background-color:#0f172a;
}

.main-title{
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#94a3b8;
    margin-bottom:25px;
}

.user-msg{
    background:#2563eb;
    color:white;
    padding:12px;
    border-radius:12px;
    margin:8px;
}

.bot-msg{
    background:#1e293b;
    color:white;
    padding:12px;
    border-radius:12px;
    margin:8px;
    border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

with open("intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern.lower())
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

# ---------------- BOT RESPONSE ----------------

def get_response(user_input):

    user_input = user_input.lower()

    user_vector = vectorizer.transform([user_input])

    predicted_tag = model.predict(user_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."

# ---------------- SESSION ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🎓 College Assistant")

    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("Quick Questions")

    if st.button("💰 Fees"):
        q = "fees"
        st.session_state.messages.append(
            {"role":"user","content":q}
        )
        st.session_state.messages.append(
            {"role":"bot","content":get_response(q)}
        )
        st.rerun()

    if st.button("📝 Admission"):
        q = "admission"
        st.session_state.messages.append(
            {"role":"user","content":q}
        )
        st.session_state.messages.append(
            {"role":"bot","content":get_response(q)}
        )
        st.rerun()

    if st.button("🏢 Placement"):
        q = "placement"
        st.session_state.messages.append(
            {"role":"user","content":q}
        )
        st.session_state.messages.append(
            {"role":"bot","content":get_response(q)}
        )
        st.rerun()

    if st.button("📚 Courses"):
        q = "courses"
        st.session_state.messages.append(
            {"role":"user","content":q}
        )
        st.session_state.messages.append(
            {"role":"bot","content":get_response(q)}
        )
        st.rerun()

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🎓 Chatbot for College Queries</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Ask anything about fees, admission, placement, courses and more</div>',
    unsafe_allow_html=True
)

# ---------------- CHAT HISTORY ----------------

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f'<div class="user-msg">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div class="bot-msg">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ---------------- CHAT INPUT ----------------

user_input = st.chat_input("Type your question here...")

if user_input:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )

    response = get_response(user_input)

    st.session_state.messages.append(
        {
            "role":"bot",
            "content":response
        }
    )

    st.rerun()