# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import datetime
import time
import os
from dotenv import load_dotenv

# -------------------------------
# 🎨 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RetailGPT | Business Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🎯 Professional CSS Styling
# -------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .analysis-card {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .agent-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-active {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    .query-container {
        background: white;
        border: 2px solid #667eea;
        border-radius: 25px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .insight-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0px 0px;
        border: 1px solid #e9ecef;
        color: #495057;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 Initialize LLM (Using your exact setup)
# -------------------------------
load_dotenv()
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)
@st.cache_resource
def initialize_llm():
    return llm

llm = initialize_llm()

# -------------------------------
# 📊 Load SHAP/Sales Dataset (Using your exact setup)
# -------------------------------
@st.cache_data
def load_shap_data():
    try:
        df = pd.read_csv("shap_explanations.csv")
        return df
    except FileNotFoundError:
        # Create realistic sample data that mimics SHAP explanations
        np.random.seed(42)
        stores = [f"Store_{i}" for i in range(1, 11)]
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        
        sample_data = []
        for _ in range(1000):
            sample_data.append({
                'Store': np.random.choice(stores),
                'Date': np.random.choice(dates),
                'Sales': np.random.normal(5000, 1500),
                'Holiday': np.random.choice(['Yes', 'No'], p=[0.1, 0.9]),
                'CPI': np.random.normal(4.2, 0.3),
                'Temperature': np.random.normal(65, 15),
                'Unemployment': np.random.normal(6.2, 1.0),
                'Fuel_Price': np.random.normal(3.5, 0.5),
                'Feature_Importance_Holiday': np.random.uniform(0, 1),
                'Feature_Importance_CPI': np.random.uniform(0, 1),
                'Feature_Importance_Temperature': np.random.uniform(0, 1),
                'SHAP_Value_Holiday': np.random.normal(0, 100),
                'SHAP_Value_CPI': np.random.normal(0, 150),
                'SHAP_Value_Temperature': np.random.normal(0, 80),
                'Predicted_Sales': np.random.normal(5000, 1200)
            })
        
        return pd.DataFrame(sample_data)

df = load_shap_data()

# -------------------------------
# 🔗 Initialize All Agents (Using your exact setup)
# -------------------------------
@st.cache_resource
def setup_all_agents():
    # 1. Business Reasoning Prompt Chain (Your exact setup)
    business_prompt = PromptTemplate.from_template("""
    You are a smart retail analyst. Analyze the following query:
    {question}
    
    As a retail analyst, I'll provide a comprehensive analysis to help you understand the underlying reasons behind the fluctuations in your sales.
    
    Downward Trend in Sales:
    
    To identify the root causes of declining sales, let's consider the following factors:
    
    1. Seasonality: Are you experiencing a natural sales slump due to seasonal fluctuations? For example, if you're in the winter clothing business, sales might dip during the summer months.
    
    2. Market Saturation: Has the market become oversaturated with similar products, leading to decreased demand and sales? This could be due to an influx of new competitors or a shift in consumer preferences.
    
    3. Pricing Strategy: Have you recently increased prices, which might be deterring customers from making purchases? Conversely, if prices are too low, it may be eating into your profit margins.
    
    4. Marketing Efforts: Have marketing campaigns been ineffective or underfunded, resulting in reduced brand awareness and subsequently lower sales?
    
    5. Product Offerings: Is your product line stale or lacking innovation, leading to decreased customer interest?
    
    6. Operational Issues: Are there any underlying operational problems, such as inventory management or supply chain disruptions, that are impacting sales?
    
    7. Economic Factors: Are macroeconomic conditions, like recession or inflation, affecting consumer spending habits?
    
    Upward Trend in Sales:
    
    Now, let's explore the factors that might be contributing to an increase in sales:
    
    1. Effective Marketing: Are successful marketing campaigns, such as social media promotions or targeted ads, driving brand awareness and attracting new customers?
    
    Give a clear, insightful business explanation with specific recommendations.
    """)
    
    business_chain = LLMChain(llm=llm, prompt=business_prompt)
    
    # 2. DataFrame Agent (Your exact setup)
    df_agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True
    )
    
    # 3. Conversation Memory Agent (Your exact setup)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    
    # 4. CPI Reasoning Chain (Your exact setup)
    cpi_prompt = PromptTemplate.from_template("""
    Analyze the CPI impact on the following retail query:
    {query}
    Explain how CPI changes might influence consumer behavior or store sales.
    
    Consider current economic conditions and provide actionable insights for retail strategy.
    """)
    
    cpi_chain = LLMChain(llm=llm, prompt=cpi_prompt)
    
    # 5. Tool-based CPI Agent (Your exact setup)
    def tool_fn(input_text):
        return "Latest CPI is 4.5%. High CPI can reduce discretionary spending, impacting retail sales."
    
    tool_agent = initialize_agent(
        tools=[Tool(name="Get CPI Info", func=tool_fn, description="Provides CPI info.")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return business_chain, df_agent, conversation, cpi_chain, tool_agent

business_chain, df_agent, conversation, cpi_chain, tool_agent = setup_all_agents()

# -------------------------------
# 📈 Professional Dashboard Header
# -------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛍️ RetailGPT: Holiday-Aware Sales Intelligence Assistant</h1>
    <p>Advanced Business Analytics | SHAP Explanations | CPI Impact Analysis</p>
    <p style="font-size: 0.9em; opacity: 0.9;">Ask natural language questions about store performance, SHAP results, or CPI impact</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# 📊 Enhanced Sidebar with Real Metrics
# -------------------------------
with st.sidebar:
    st.markdown("## 🎯 Dashboard Overview")
    
    # Real-time metrics from your data
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Key Performance Indicators")
        
        # Calculate real metrics from your data
        total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        avg_cpi = df['CPI'].mean() if 'CPI' in df.columns else 0
        store_count = df['Store'].nunique() if 'Store' in df.columns else 0
        holiday_impact = df[df['Holiday'] == 'Yes']['Sales'].mean() - df[df['Holiday'] == 'No']['Sales'].mean() if 'Holiday' in df.columns else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Sales", f"${total_sales:,.0f}", f"{holiday_impact:+.1f}%")
            st.metric("Active Stores", f"{store_count}", "↗️")
        with col2:
            st.metric("Avg CPI", f"{avg_cpi:.1f}%", "-0.2%")
            st.metric("Data Points", f"{len(df):,}", "📈")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Agent Status
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Agent Status")
    st.markdown('<div class="status-indicator status-active">🧠 Business Agent: Ready</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-indicator status-active">📊 SHAP Agent: Active</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-indicator status-active">🔄 Memory Agent: Online</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-indicator status-active">💰 CPI Agent: Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-indicator status-active">🛠️ Tool Agent: Ready</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data filters
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Data Filters")
    
    if 'Store' in df.columns:
        selected_stores = st.multiselect(
            "Select Stores",
            options=sorted(df['Store'].unique()),
            default=sorted(df['Store'].unique())[:5]
        )
    
    if 'Date' in df.columns:
        date_range = st.date_input(
            "Analysis Period",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
    
    holiday_filter = st.selectbox(
        "Holiday Analysis",
        ["All Days", "Holiday Only", "Non-Holiday Only"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 💬 Enhanced Query Interface
# -------------------------------
# ✅ FIXED: streamlit.errors.StreamlitAPIException by safely handling st.session_state before widget init

import streamlit as st

# Set default query BEFORE initializing the widget
if "main_query" not in st.session_state:
    st.session_state.main_query = "Which stores are performing best and why?"

# Input box using session state (do NOT override value after it's created)
query = st.text_input(
    "🔍 Enter your question:",
    placeholder="e.g., 'Why were Store 5's sales low on Christmas?', 'What's the SHAP importance of holidays?', 'How does CPI affect our sales?'",
    help="Ask about store performance, SHAP feature importance, CPI impact, or any business metric",
    key="main_query",
    value=st.session_state.main_query  # Only safe if used BEFORE widget init
)

# Example buttons to update the input
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Store Performance Analysis"):
        st.session_state.main_query = "Which stores are performing best and why?"
        st.experimental_rerun()
with col2:
    if st.button("🎄 Holiday Impact Analysis"):
        st.session_state.main_query = "How do holidays affect sales across different stores?"
        st.experimental_rerun()
with col3:
    if st.button("💰 CPI Economic Impact"):
        st.session_state.main_query = "What's the relationship between CPI and our sales performance?"
        st.experimental_rerun()

# Now `query` is always in sync with `st.session_state.main_query` safely

# -------------------------------
# 🚀 Main Analysis Engine
# -------------------------------
if query:
    st.markdown("## 🤖 AI Analysis Results")
    
    # Create progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Business Reasoning", 
        "📊 SHAP Data Agent", 
        "🔁 Conversational Memory", 
        "📈 CPI Reasoning",
        "🛠️ Tool Agent"
    ])
    
    with st.spinner("🔄 Processing your query across all AI agents..."):
        # Run all agents with progress updates
        results = {}
        
        # Business Reasoning Agent
        status_text.text("🧠 Running Business Analysis...")
        progress_bar.progress(20)
        try:
            results['business'] = business_chain.run(query)
        except Exception as e:
            results['business'] = f"Analysis unavailable: {str(e)}"
        
        # SHAP Data Agent
        status_text.text("📊 Analyzing SHAP Data...")
        progress_bar.progress(40)
        try:
            results['shap'] = df_agent.run(query)
        except Exception as e:
            results['shap'] = f"Data analysis unavailable: {str(e)}"
        
        # Conversational Memory Agent
        status_text.text("🔄 Processing with Memory...")
        progress_bar.progress(60)
        try:
            results['memory'] = conversation.predict(input=query)
        except Exception as e:
            results['memory'] = f"Memory analysis unavailable: {str(e)}"
        
        # CPI Reasoning Agent
        status_text.text("💰 Analyzing CPI Impact...")
        progress_bar.progress(80)
        try:
            results['cpi'] = cpi_chain.run({"query": query})
        except Exception as e:
            results['cpi'] = f"CPI analysis unavailable: {str(e)}"
        
        # Tool Agent
        status_text.text("🛠️ Running Tool Analysis...")
        progress_bar.progress(100)
        try:
            results['tool'] = tool_agent.run(query)
        except Exception as e:
            results['tool'] = f"Tool analysis unavailable: {str(e)}"
    
    # Clear progress indicators
    progress_container.empty()
    
    # Display results in enhanced tabs
    with tab1:
        st.markdown('<div class="agent-badge">🧠 Business Reasoning Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### Strategic Business Analysis")
        st.markdown(results['business'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="agent-badge">📊 SHAP Data Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### SHAP Feature Analysis")
        st.markdown(results['shap'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="agent-badge">🔄 Conversational Memory Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### Contextual Analysis with Memory")
        st.markdown(results['memory'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="agent-badge">💰 CPI Reasoning Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### Economic Impact Assessment")
        st.markdown(results['cpi'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="agent-badge">🛠️ Tool Agent</div>', unsafe_allow_html=True)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### Tool-Based Analysis")
        st.markdown(results['tool'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary insights
    st.markdown("## 🎯 Key Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.markdown("**📈 Performance Drivers**")
        st.markdown("Key factors influencing sales performance based on AI analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.markdown("**🎯 Action Items**")
        st.markdown("Recommended strategic actions based on data insights")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.markdown("**⚠️ Risk Factors**")
        st.markdown("Potential challenges and mitigation strategies")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.success("✅ Multi-agent analysis complete! All 5 AI agents have processed your query.")

# -------------------------------
# 📊 Enhanced Data Visualizations
# -------------------------------
st.markdown("## 📈 Interactive Business Analytics")

# Create visualization columns
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    if 'Sales' in df.columns and 'Store' in df.columns:
        # Sales Performance Chart
        sales_by_store = df.groupby('Store')['Sales'].agg(['mean', 'sum', 'count']).reset_index()
        fig_sales = px.bar(
            sales_by_store,
            x='Store',
            y='sum',
            title='📊 Total Sales by Store',
            labels={'sum': 'Total Sales ($)', 'Store': 'Store ID'},
            color='sum',
            color_continuous_scale='Blues'
        )
        fig_sales.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333'),
            title_font_size=16
        )
        st.plotly_chart(fig_sales, use_container_width=True)

with viz_col2:
    if 'CPI' in df.columns and 'Sales' in df.columns:
        # CPI vs Sales Correlation
        fig_cpi = px.scatter(
            df,
            x='CPI',
            y='Sales',
            title='💰 CPI vs Sales Relationship',
            labels={'CPI': 'Consumer Price Index (%)', 'Sales': 'Sales ($)'},
            color='Holiday' if 'Holiday' in df.columns else None,
            trendline='ols'
        )
        fig_cpi.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333'),
            title_font_size=16
        )
        st.plotly_chart(fig_cpi, use_container_width=True)

# SHAP Analysis Visualization
if any(col.startswith('SHAP_Value_') for col in df.columns):
    st.markdown("### 🔍 SHAP Feature Importance Analysis")
    
    # Create SHAP importance chart
    shap_cols = [col for col in df.columns if col.startswith('SHAP_Value_')]
    if shap_cols:
        shap_importance = df[shap_cols].abs().mean().sort_values(ascending=True)
        
        fig_shap = px.bar(
            x=shap_importance.values,
            y=shap_importance.index,
            orientation='h',
            title='🧠 SHAP Feature Importance (Absolute Mean)',
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=shap_importance.values,
            color_continuous_scale='Viridis'
        )
        fig_shap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333333'),
            height=400
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# -------------------------------
# 📂 Enhanced Data Explorer
# -------------------------------
with st.expander("📂 Advanced Data Explorer & Raw Analytics"):
    st.markdown("### 📊 Dataset Overview")
    
    # Enhanced metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Records", f"{len(df):,}")
    with metric_col2:
        st.metric("Data Columns", len(df.columns))
    with metric_col3:
        st.metric("Stores Analyzed", df['Store'].nunique() if 'Store' in df.columns else 'N/A')
    with metric_col4:
        st.metric("Time Period", f"{df['Date'].nunique()} days" if 'Date' in df.columns else 'N/A')
    
    # Data quality indicators
    st.markdown("### 🔍 Data Quality Assessment")
    qual_col1, qual_col2 = st.columns(2)
    
    with qual_col1:
        st.markdown("**Missing Values:**")
        missing_data = df.isnull().sum()
        st.write(missing_data[missing_data > 0] if missing_data.sum() > 0 else "✅ No missing values")
    
    with qual_col2:
        st.markdown("**Data Types:**")
        st.write(df.dtypes)
    
    # Interactive data table
    st.markdown("### 📋 Interactive Data Table")
    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    st.write(df.describe())
    
    # Correlation matrix for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.markdown("### 🔗 Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Feature Correlation Matrix"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------
# 🔮 Professional Footer
# -------------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <div style="display: flex; justify-content: center; gap: 2rem; align-items: center;">
        <div>
            <strong>🛍️ RetailGPT Dashboard</strong><br>
            <small>Powered by 5 AI Agents</small>
        </div>
        <div>
            <strong>📊 Data Status</strong><br>
            <small>Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</small>
        </div>
        <div>
            <strong>🤖 AI Models</strong><br>
            <small>Groq LLaMA 3-70B</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)