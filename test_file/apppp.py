# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import datetime
import numpy as np
from typing import Dict, List
import json
import os
from dotenv import load_dotenv
# -------------------------------
# 🎨 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RetailGPT Business Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .stTabs > div > div > div > div {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 Initialize LLM and Chains
# -------------------------------

load_dotenv()
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)
@st.cache_resource
def initialize_llm():
    """Initialize the LLM with caching for better performance"""
    return llm

@st.cache_resource
def setup_chains(_llm):
    """Setup all LangChain components with caching"""
    
    # Enhanced Business Reasoning Chain
    business_prompt = PromptTemplate.from_template("""
    You are a senior retail business analyst with expertise in consumer behavior and market trends.
    
    Context: Analyze this retail business query with focus on:
    - Key performance indicators (KPIs)
    - Consumer behavior patterns
    - Market trends and seasonality
    - Competitive analysis insights
    - Actionable recommendations
    
    Query: {question}
    
    Provide a comprehensive business analysis with:
    1. Key insights
    2. Potential root causes
    3. Actionable recommendations
    4. Risk factors to consider
    """)
    
    # SHAP Explanation Chain
    shap_prompt = PromptTemplate.from_template("""
    You are an expert in machine learning interpretability and SHAP (SHapley Additive exPlanations).
    
    Analyze the following query related to feature importance and model predictions:
    {question}
    
    Provide insights on:
    1. Feature importance rankings
    2. Model prediction explanations
    3. Business implications of key features
    4. Recommendations based on SHAP values
    """)
    
    # CPI Impact Chain
    cpi_prompt = PromptTemplate.from_template("""
    You are a retail economist specializing in Consumer Price Index (CPI) impact analysis.
    
    Current Economic Context:
    - CPI trends and inflation rates
    - Consumer spending patterns
    - Retail sector impacts
    
    Query: {query}
    
    Analyze how CPI changes affect:
    1. Consumer purchasing power
    2. Demand patterns across categories
    3. Pricing strategies
    4. Inventory management
    5. Sales forecasting adjustments
    """)
    
    # Initialize chains
    business_chain = LLMChain(llm=_llm, prompt=business_prompt)
    shap_chain = LLMChain(llm=_llm, prompt=shap_prompt)
    cpi_chain = LLMChain(llm=_llm, prompt=cpi_prompt)
    
    # Conversation memory
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=_llm, memory=memory, verbose=False)
    
    return business_chain, shap_chain, cpi_chain, conversation

# -------------------------------
# 📊 Data Loading and Processing
# -------------------------------
@st.cache_data
def load_and_process_data():
    """Load and process the retail dataset"""
    try:
        # Try to load the main dataset
        df = pd.read_csv("shap_explanations.csv")
        
        # Handle different possible column names
        column_mapping = {
            'Sales': 'Weekly_Sales',
            'Store_ID': 'Store',
            'Holiday': 'Holiday_Flag',
            'Customers': 'Size'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        st.warning("Data file not found. Using sample data for demonstration.")
        df = create_sample_data()
    
    # Data processing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure we have a sales column
    if 'Weekly_Sales' not in df.columns and 'Sales' in df.columns:
        df['Weekly_Sales'] = df['Sales']
    
    # Ensure we have a store column
    if 'Store' not in df.columns and 'Store_ID' in df.columns:
        df['Store'] = df['Store_ID']
    
    return df

def create_sample_data():
    """Create sample retail data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='W')  # Weekly data
    
    data = {
        'Date': dates,
        'Store': np.random.randint(1, 11, len(dates)),
        'Weekly_Sales': np.random.normal(50000, 15000, len(dates)),
        'Size': np.random.normal(150000, 50000, len(dates)),
        'Temperature': np.random.normal(70, 20, len(dates)),
        'CPI': np.random.normal(240, 10, len(dates)),
        'Holiday_Flag': np.random.choice([0, 1], len(dates), p=[0.9, 0.1]),
        'Fuel_Price': np.random.normal(3.5, 0.5, len(dates)),
        'Unemployment': np.random.normal(7.5, 2.0, len(dates)),
        'Dept': np.random.randint(1, 100, len(dates)),
        'IsHoliday': np.random.choice([True, False], len(dates), p=[0.1, 0.9])
    }
    
    df = pd.DataFrame(data)
    df['Weekly_Sales'] = np.where(df['Holiday_Flag'] == 1, df['Weekly_Sales'] * 1.5, df['Weekly_Sales'])
    df['Weekly_Sales'] = np.where(df['IsHoliday'] == True, df['Weekly_Sales'] * 1.3, df['Weekly_Sales'])
    
    return df

# -------------------------------
# 📈 Visualization Functions
# -------------------------------
def create_kpi_metrics(df):
    """Create KPI metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Determine which sales column to use
    sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Sales'
    store_col = 'Store' if 'Store' in df.columns else 'Store_ID'
    
    with col1:
        total_sales = df[sales_col].sum()
        st.metric(
            label="Total Sales",
            value=f"${total_sales:,.0f}",
            delta=f"{(total_sales/1000000):.1f}M"
        )
    
    with col2:
        avg_sales = df[sales_col].mean()
        st.metric(
            label="Average Weekly Sales",
            value=f"${avg_sales:,.0f}",
            delta="5.2%"
        )
    
    with col3:
        if 'Size' in df.columns:
            avg_size = df['Size'].mean()
            st.metric(
                label="Average Store Size",
                value=f"{avg_size:,.0f} sq ft",
                delta="2.1%"
            )
        elif 'Customers' in df.columns:
            total_customers = df['Customers'].sum()
            st.metric(
                label="Total Customers",
                value=f"{total_customers:,.0f}",
                delta="2.1%"
            )
        else:
            unique_stores = df[store_col].nunique()
            st.metric(
                label="Number of Stores",
                value=f"{unique_stores}",
                delta="Active"
            )
    
    with col4:
        if 'Unemployment' in df.columns:
            avg_unemployment = df['Unemployment'].mean()
            st.metric(
                label="Avg Unemployment Rate",
                value=f"{avg_unemployment:.1f}%",
                delta="-0.3%"
            )
        elif 'CPI' in df.columns:
            avg_cpi = df['CPI'].mean()
            st.metric(
                label="Average CPI",
                value=f"{avg_cpi:.1f}",
                delta="1.8%"
            )
        else:
            conversion_rate = df[sales_col].std()
            st.metric(
                label="Sales Volatility",
                value=f"${conversion_rate:,.0f}",
                delta="Std Dev"
            )

def create_sales_trend_chart(df):
    """Create sales trend visualization"""
    sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Sales'
    
    if 'Date' in df.columns:
        daily_sales = df.groupby('Date')[sales_col].sum().reset_index()
        
        fig = px.line(
            daily_sales, 
            x='Date', 
            y=sales_col,
            title='📈 Sales Trend Over Time',
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Date",
            yaxis_title=f"{sales_col.replace('_', ' ')} ($)"
        )
        return fig
    return None

def create_store_performance_chart(df):
    """Create store performance comparison"""
    sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Sales'
    store_col = 'Store' if 'Store' in df.columns else 'Store_ID'
    
    if store_col in df.columns:
        store_sales = df.groupby(store_col)[sales_col].sum().reset_index()
        
        fig = px.bar(
            store_sales,
            x=store_col,
            y=sales_col,
            title='🏪 Store Performance Comparison',
            color=sales_col,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=400,
            xaxis_title="Store",
            yaxis_title=f"{sales_col.replace('_', ' ')} ($)"
        )
        return fig
    return None

def create_category_analysis(df):
    """Create category performance analysis"""
    sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Sales'
    
    # Check for different category columns
    category_col = None
    for col in ['Category', 'Dept', 'Department']:
        if col in df.columns:
            category_col = col
            break
    
    if category_col:
        category_sales = df.groupby(category_col)[sales_col].sum().reset_index()
        
        fig = px.pie(
            category_sales,
            values=sales_col,
            names=category_col,
            title=f'📊 Sales by {category_col}',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        return fig
    
    # If no category column, create holiday vs non-holiday analysis
    elif 'Holiday_Flag' in df.columns:
        holiday_sales = df.groupby('Holiday_Flag')[sales_col].sum().reset_index()
        holiday_sales['Holiday_Flag'] = holiday_sales['Holiday_Flag'].map({0: 'Non-Holiday', 1: 'Holiday'})
        
        fig = px.pie(
            holiday_sales,
            values=sales_col,
            names='Holiday_Flag',
            title='📊 Holiday vs Non-Holiday Sales',
            color_discrete_sequence=['#ff7f0e', '#1f77b4']
        )
        fig.update_layout(height=400)
        return fig
    
    return None

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title='🔗 Feature Correlation Matrix',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=500)
    return fig

# -------------------------------
# 🤖 AI Analysis Functions
# -------------------------------
def run_comprehensive_analysis(query: str, df: pd.DataFrame, chains: tuple):
    """Run comprehensive AI analysis"""
    business_chain, shap_chain, cpi_chain, conversation = chains
    
    # Create DataFrame agent
    df_agent = create_pandas_dataframe_agent(
        initialize_llm(), df, verbose=False, allow_dangerous_code=True
    )
    
    with st.spinner("🔍 Running comprehensive analysis..."):
        # Run different types of analysis
        results = {}
        
        try:
            results['business'] = business_chain.run(query)
        except Exception as e:
            results['business'] = f"Analysis unavailable: {str(e)}"
        
        try:
            results['data'] = df_agent.run(query)
        except Exception as e:
            results['data'] = f"Data analysis unavailable: {str(e)}"
        
        try:
            results['conversation'] = conversation.predict(input=query)
        except Exception as e:
            results['conversation'] = f"Conversation analysis unavailable: {str(e)}"
        
        try:
            results['cpi'] = cpi_chain.run({"query": query})
        except Exception as e:
            results['cpi'] = f"CPI analysis unavailable: {str(e)}"
    
    return results

# -------------------------------
# 🚀 Main Application
# -------------------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🛍️ RetailGPT Business Intelligence Platform</h1>
        <p style="color: #e0e0e0; margin: 0.5rem 0 0 0;">Advanced AI-Powered Retail Analytics & Insights Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    llm = initialize_llm()
    chains = setup_chains(llm)
    df = load_and_process_data()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Focus",
            ["Comprehensive", "Business Strategy", "Data Insights", "Economic Impact", "SHAP Analysis"]
        )
        
        # Date range filter
        if 'Date' in df.columns:
            date_range = st.date_input(
                "Date Range",
                value=(df['Date'].min(), df['Date'].max()),
                min_value=df['Date'].min(),
                max_value=df['Date'].max()
            )
        
        # Store filter
        store_col = 'Store' if 'Store' in df.columns else 'Store_ID'
        if store_col in df.columns:
            stores = st.multiselect(
                "Select Stores",
                options=sorted(df[store_col].unique()),
                default=sorted(df[store_col].unique())
            )
        
        # Department/Category filter
        dept_col = None
        for col in ['Dept', 'Department', 'Category']:
            if col in df.columns:
                dept_col = col
                break
        
        if dept_col:
            departments = st.multiselect(
                f"Select {dept_col}s",
                options=sorted(df[dept_col].unique()),
                default=sorted(df[dept_col].unique())[:5]  # Show only first 5 by default
            )
        
        # Holiday filter
        holiday_col = None
        for col in ['Holiday_Flag', 'IsHoliday', 'Holiday']:
            if col in df.columns:
                holiday_col = col
                break
        
        if holiday_col:
            holiday_options = ["All", "Holiday", "Non-Holiday"]
            holiday_filter = st.selectbox("Holiday Filter", holiday_options)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Records", len(df))
        
        store_col = 'Store' if 'Store' in df.columns else 'Store_ID'
        if store_col in df.columns:
            st.metric("Stores", df[store_col].nunique())
        
        if 'Date' in df.columns:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        
        # Additional metrics
        sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Sales'
        if sales_col in df.columns:
            st.metric("Avg Sales", f"${df[sales_col].mean():,.0f}")
        
        if 'CPI' in df.columns:
            st.metric("Avg CPI", f"{df['CPI'].mean():.1f}")
        
        if 'Temperature' in df.columns:
            st.metric("Avg Temp", f"{df['Temperature'].mean():.1f}°F")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 AI Analysis", "📈 Deep Dive", "🔍 Data Explorer"])
    
    with tab1:
        st.header("📊 Business Dashboard")
        
        # KPI Metrics
        create_kpi_metrics(df)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            sales_chart = create_sales_trend_chart(df)
            if sales_chart:
                st.plotly_chart(sales_chart, use_container_width=True)
        
        with col2:
            store_chart = create_store_performance_chart(df)
            if store_chart:
                st.plotly_chart(store_chart, use_container_width=True)
        
        # Additional charts
        col3, col4 = st.columns(2)
        
        with col3:
            category_chart = create_category_analysis(df)
            if category_chart:
                st.plotly_chart(category_chart, use_container_width=True)
        
        with col4:
            correlation_chart = create_correlation_heatmap(df)
            if correlation_chart:
                st.plotly_chart(correlation_chart, use_container_width=True)
    
    with tab2:
        st.header("🤖 AI-Powered Analysis")
        
        # Query input
        query = st.text_area(
            "💬 Ask your business question:",
            placeholder="e.g., 'Why were Store 5's sales low during the holiday season?' or 'What factors most influence customer purchasing behavior?'",
            height=100
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if query:
                results = run_comprehensive_analysis(query, df, chains)
                
                # Display results in expandable sections
                with st.expander("🧠 Business Strategy Analysis", expanded=True):
                    st.markdown(f"<div class='insight-box'>{results['business']}</div>", unsafe_allow_html=True)
                
                with st.expander("📊 Data Intelligence", expanded=True):
                    st.markdown(f"<div class='insight-box'>{results['data']}</div>", unsafe_allow_html=True)
                
                with st.expander("💬 Conversational Context", expanded=False):
                    st.markdown(f"<div class='insight-box'>{results['conversation']}</div>", unsafe_allow_html=True)
                
                with st.expander("📈 Economic Impact Analysis", expanded=False):
                    st.markdown(f"<div class='insight-box'>{results['cpi']}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a question to analyze.")
    
    with tab3:
        st.header("📈 Deep Dive Analytics")
        
        # Advanced analytics options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Performance Metrics")
            
            sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Weekly_Sales'
            if sales_col in df.columns:
                # Sales distribution
                fig = px.histogram(df, x=sales_col, nbins=50, title=f'{sales_col.replace("_", " ")} Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Sales vs other factors
                if 'Temperature' in df.columns:
                    fig_scatter = px.scatter(df, x='Temperature', y=sales_col, 
                                           title=f'{sales_col.replace("_", " ")} vs Temperature',
                                           trendline="ols")
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("📊 Trend Analysis")
            
            if 'Date' in df.columns and sales_col in df.columns:
                # Monthly trends
                df_temp = df.copy()
                df_temp['Month'] = df_temp['Date'].dt.to_period('M')
                monthly_sales = df_temp.groupby('Month')[sales_col].sum().reset_index()
                monthly_sales['Month'] = monthly_sales['Month'].astype(str)
                
                fig = px.line(monthly_sales, x='Month', y=sales_col, 
                            title=f'Monthly {sales_col.replace("_", " ")} Trend')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Holiday impact analysis
                holiday_col = None
                for col in ['Holiday_Flag', 'IsHoliday', 'Holiday']:
                    if col in df.columns:
                        holiday_col = col
                        break
                
                if holiday_col:
                    holiday_impact = df.groupby(holiday_col)[sales_col].mean().reset_index()
                    if holiday_col == 'Holiday_Flag':
                        holiday_impact[holiday_col] = holiday_impact[holiday_col].map({0: 'Non-Holiday', 1: 'Holiday'})
                    elif holiday_col == 'IsHoliday':
                        holiday_impact[holiday_col] = holiday_impact[holiday_col].map({False: 'Non-Holiday', True: 'Holiday'})
                    
                    fig_holiday = px.bar(holiday_impact, x=holiday_col, y=sales_col,
                                       title=f'Holiday Impact on {sales_col.replace("_", " ")}')
                    st.plotly_chart(fig_holiday, use_container_width=True)
    
    with tab4:
        st.header("🔍 Data Explorer")
        
        # Data filtering options
        col1, col2, col3 = st.columns(3)
        
        store_col = 'Store' if 'Store' in df.columns else 'Store_ID'
        sales_col = 'Weekly_Sales' if 'Weekly_Sales' in df.columns else 'Sales'
        
        with col1:
            if store_col in df.columns:
                store_filter = st.multiselect("Filter by Store", df[store_col].unique())
        
        with col2:
            # Check for category/department columns
            category_col = None
            for col in ['Category', 'Dept', 'Department']:
                if col in df.columns:
                    category_col = col
                    break
            
            if category_col:
                category_filter = st.multiselect(f"Filter by {category_col}", df[category_col].unique())
        
        with col3:
            # Check for holiday columns
            holiday_col = None
            for col in ['Holiday_Flag', 'IsHoliday', 'Holiday']:
                if col in df.columns:
                    holiday_col = col
                    break
            
            if holiday_col:
                holiday_filter = st.selectbox("Holiday Filter", ["All", "Holiday", "Non-Holiday"])
        
        # Apply filters
        filtered_df = df.copy()
        
        if store_col in df.columns and 'store_filter' in locals() and store_filter:
            filtered_df = filtered_df[filtered_df[store_col].isin(store_filter)]
        
        if category_col and 'category_filter' in locals() and category_filter:
            filtered_df = filtered_df[filtered_df[category_col].isin(category_filter)]
        
        if holiday_col and 'holiday_filter' in locals() and holiday_filter != "All":
            if holiday_col == 'Holiday_Flag':
                holiday_value = 1 if holiday_filter == "Holiday" else 0
                filtered_df = filtered_df[filtered_df[holiday_col] == holiday_value]
            elif holiday_col == 'IsHoliday':
                holiday_value = True if holiday_filter == "Holiday" else False
                filtered_df = filtered_df[filtered_df[holiday_col] == holiday_value]
        
        # Display filtered data
        st.subheader("📋 Filtered Dataset")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        st.write(filtered_df.describe())
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"retail_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🛍️ RetailGPT Business Intelligence Platform | Powered by AI & Advanced Analytics</p>
        <p>Built with Streamlit, LangChain, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()