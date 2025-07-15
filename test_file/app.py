# app.py

import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv
# -------------------------------
# 🧠 Initialize LLM
# -------------------------------
load_dotenv()
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------
# 📊 Load SHAP/Sales Dataset
# -------------------------------
df = pd.read_csv("shap_explanations.csv")  # or Walmart_Sales.csv

# -------------------------------
# 🔗 Chains Setup
# -------------------------------

# 1. Business Reasoning Prompt Chain
business_prompt = PromptTemplate.from_template("""
You are a smart retail analyst. Analyze the following query:
{question}

Give a clear, insightful business explanation.
""")
business_chain = LLMChain(llm=llm, prompt=business_prompt)

# 2. DataFrame Agent
df_agent = create_pandas_dataframe_agent(
    llm, df, verbose=True, allow_dangerous_code=True
)

# 3. Conversation Memory Agent
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# 4. CPI Reasoning Chain
cpi_prompt = PromptTemplate.from_template("""
Analyze the CPI impact on the following retail query:
{query}

Explain how CPI changes might influence consumer behavior or store sales.
""")
cpi_chain = LLMChain(llm=llm, prompt=cpi_prompt)

# 5. Tool-based CPI Agent (optional)
def tool_fn(input_text):
    return "Latest CPI is 4.5%. High CPI can reduce discretionary spending, impacting retail sales."

tool_agent = initialize_agent(
    tools=[Tool(name="Get CPI Info", func=tool_fn, description="Provides CPI info.")],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# -------------------------------
# 🚀 Streamlit UI
# -------------------------------
st.set_page_config(page_title="RetailGPT Dashboard", layout="wide")
st.title("🛍️ RetailGPT: Holiday-Aware Sales Intelligence Assistant")
st.markdown("Ask natural language questions about store performance, SHAP results, or CPI impact.")

query = st.text_input("💬 Ask your question (e.g. 'Why were Store 5's sales low on Christmas?')")

if query:
    with st.spinner("🤖 Analyzing..."):
        # Run all agents
        business_resp = business_chain.run(query)
        df_resp = df_agent.run(query)
        convo_resp = conversation.predict(input=query)
        cpi_resp = cpi_chain.run({"query": query})

    # Output
    st.subheader("🧠 Business Reasoning")
    st.markdown(business_resp)

    st.subheader("📊 SHAP Data Agent")
    st.markdown(df_resp)

    st.subheader("🔁 Conversational Memory")
    st.markdown(convo_resp)

    st.subheader("📈 CPI Reasoning")
    st.markdown(cpi_resp)

    st.success("✅ Analysis complete! Ask another query.")

# Optional: Show raw data
with st.expander("📂 View Underlying Data"):
    st.dataframe(df)
