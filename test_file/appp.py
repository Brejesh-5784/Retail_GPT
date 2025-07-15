import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample dataset (replace with real one)
sample_sales = pd.DataFrame({
    "Week": pd.date_range("2023-10-01", periods=10, freq='W'),
    "Store 2": np.random.randint(20000, 50000, 10),
    "Store 8": np.random.randint(25000, 48000, 10),
})

# Streamlit App Config
st.set_page_config(page_title="RetailGPT Assistant", layout="wide", page_icon="🛍️")

# CSS Styling
st.markdown("""
    <style>
    .chat-bubble {
        padding: 12px 18px;
        margin: 10px;
        border-radius: 20px;
        max-width: 70%;
        font-size: 16px;
    }
    .user-bubble {
        background-color: #4a90e2;
        color: white;
        margin-left: auto;
    }
    .ai-bubble {
        background-color: #50e3c2;
        color: black;
        margin-right: auto;
    }
    .header {
        font-size: 2.2em;
        text-align: center;
        color: #ffcc00;
        font-weight: bold;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🤖 RetailGPT: Your AI Sales Chatbot</div>', unsafe_allow_html=True)

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat input
with st.form("chat-form", clear_on_submit=True):
    query = st.text_input("Ask your question:", placeholder="e.g., Compare Store 2 and Store 8")
    submitted = st.form_submit_button("Send")

# If user submits query
if submitted and query:
    st.session_state.chat.append(("user", query))

    # Fake LLM response logic
    if "compare" in query.lower() and "store 2" in query.lower() and "store 8" in query.lower():
        ai_text = "Here is a weekly sales comparison between Store 2 and Store 8 📊"
        st.session_state.chat.append(("ai", ai_text))
        st.session_state.chat.append(("chart", "comparison"))

    else:
        st.session_state.chat.append(("ai", f"I'm analyzing your question: **{query}**"))

# Display chat history
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">{msg}</div>', unsafe_allow_html=True)
    elif role == "ai":
        st.markdown(f'<div class="chat-bubble ai-bubble">{msg}</div>', unsafe_allow_html=True)
    elif role == "chart":
        st.markdown("#### 📊 Visual Sales Comparison")
        fig, ax = plt.subplots()
        ax.plot(sample_sales["Week"], sample_sales["Store 2"], label="Store 2", marker="o")
        ax.plot(sample_sales["Week"], sample_sales["Store 8"], label="Store 8", marker="o")
        ax.set_xlabel("Week")
        ax.set_ylabel("Sales ($)")
        ax.set_title("Weekly Sales: Store 2 vs Store 8")
        ax.legend()
        st.pyplot(fig)

# Sidebar tools
with st.sidebar:
    st.title("🛠️ Tools")
    st.info("Try asking:\n- `Compare Store 2 and Store 8`\n- `Trend of CPI vs Sales`\n- `Why were sales low on Diwali?`")

    if st.button("📥 Download Report"):
        st.success("Mock report download started! (implement actual logic)")
