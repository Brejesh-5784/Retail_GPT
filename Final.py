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
import plotly.express as px

# 🔐 Load Environment
load_dotenv()
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3, max_tokens=512
)

# 📊 Load Data
df = pd.read_csv("shap_explanations.csv")

# Prompt Chain
business_prompt = PromptTemplate.from_template("""
You are a smart retail analyst assistant working with SHAP-based retail data, economic indicators like CPI, and store-level sales insights.

Only answer if the query is business-related. Otherwise, respond with:
"This query contains inappropriate or off-topic content. Please rephrase respectfully."

If valid, answer in this format:
Summary: <20 word max sentence>
Explanation: <Clear English explanation>

Use delimiter >>> between summary and explanation.

Query: {question}
""")
business_chain = LLMChain(llm=llm, prompt=business_prompt, verbose=True)
df_agent = create_pandas_dataframe_agent(llm, df, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
cpi_prompt = PromptTemplate.from_template("""
You are an economist. Based on this retail query:
{query}

Explain how CPI may affect store sales or consumer behavior. Keep it simple and useful for retail decisions.
""")
cpi_chain = LLMChain(llm=llm, prompt=cpi_prompt, verbose=True)

# CPI Tool
def tool_fn(input_text):
    return "Current CPI is 4.5%. High CPI increases prices and can lower sales of non-essential products."

tool_agent = initialize_agent(
    tools=[Tool(name="Get CPI Info", func=tool_fn, description="Returns CPI info and its effect.")],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🎨 Custom Styling
st.set_page_config(page_title="RetailGPT AI Dashboard", layout="wide", page_icon="🛍️")

st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# 🚀 Header
st.title("🛍️ RetailGPT: AI-Powered Sales Insight Dashboard")
st.markdown("Ask questions about **store sales**, **holiday impact**, **CPI**, or **SHAP values**.")

# 📥 Query Input
query = st.text_input("🔎 Enter your business query:")

# 🧠 AI Insights Panel
if query:
    with st.spinner("🤖 Analyzing your query..."):
        business_result = business_chain.run(query)

        if ">>>" in business_result:
            short_summary, detailed_reason = business_result.split(">>>", 1)
            short_summary = short_summary.replace("Summary:", "").strip()
            detailed_reason = detailed_reason.replace("Explanation:", "").strip()
        else:
            short_summary = business_result.strip()
            detailed_reason = "No detailed explanation provided."

        df_resp = df_agent.run(query)
        convo_resp = conversation.predict(input=query)
        cpi_resp = cpi_chain.run({"query": query})

    st.markdown("## 💡 Insight Summary")
    st.success(f"**{short_summary}**")

    with st.expander("📘 Full Business Explanation"):
        st.markdown(detailed_reason)

    # 📊 Tabs for Deep Insights
    tab1, tab2, tab3 = st.tabs(["📊 SHAP Data Agent", "💬 LLM Conversation", "📈 CPI Impact"])

    with tab1:
        st.markdown(df_resp)

    with tab2:
        st.markdown(convo_resp)

    with tab3:
        st.markdown(cpi_resp)

# 📂 Data View and Download
with st.expander("📂 View & Download Dataset"):
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "retail_data.csv", "text/csv")

# 📈 Optional Graph (if column exists)
if 'Sales' in df.columns:
    fig = px.line(df, x="Date", y="Sales", title="📈 Sales Trend Over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True)
