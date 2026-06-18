import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re
import logging
from synthetic_data import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("App")

st.set_page_config(page_title="NAV Exception Dashboard", layout="wide", initial_sidebar_state="expanded")
logger.info("Initializing Streamlit Application")

from reconciliation_skill import ReconciliationSkill
from nvidia_ai_layer import RoutingDecisionEngine

st.title("NAV Exception Management - Reconciliation Skill")

import os
import pandas as pd

import glob

# Initialize state
if 'raw_data' not in st.session_state:
    data_dir = "e:\\Project\\data"
    json_files = glob.glob(os.path.join(data_dir, "*_breaks.json"))
    
    if json_files:
        df_list = []
        for file in json_files:
            try:
                df_list.append(pd.read_json(file))
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                
        if df_list:
            st.session_state.raw_data = pd.concat(df_list, ignore_index=True)
            logger.info(f"Loaded synthetic data from {len(json_files)} files in data folder.")
        else:
            st.error("Data files could not be loaded! Please run synthetic_data.py.")
            st.stop()
    else:
        st.error("No data files found! Please run synthetic_data.py to generate them.")
        st.stop()
    skill = ReconciliationSkill()
    st.session_state.processed_data = skill.process_exceptions(st.session_state.raw_data)

df = st.session_state.processed_data

# Check if we are in detailed view
if "exception_id" in st.query_params:
    selected_exception = st.query_params["exception_id"]
    if st.button("← Back to All Funds"):
        st.query_params.clear()
        st.rerun()
        
    st.title(f"Detailed View: {selected_exception}")
    matching_data = df[df['Exception ID'] == selected_exception]
    
    if len(matching_data) == 0:
        st.query_params.clear()
        st.rerun()
        
    exc_data = matching_data.iloc[0]
    
    st.markdown("---")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.info(f"**Root Cause:** {exc_data['Root Cause']}")
        st.warning(f"**Status:** {exc_data['Status']}")
        st.metric("Net Discrepancy Impact", f"${exc_data['Net Discrepancy (USD)']:,.2f}")
    with col_d2:
        st.error(f"**Possible Reason:** {exc_data['Possible Reason']}")
        st.success(f"**Resolution Recommendation:** {exc_data['Recommendation']}")
        st.metric("Fund Name", exc_data['Fund Name'])
        
    # Trend Chart
    st.subheader("Confidence Score Trend")
    trend_data = exc_data['Confidence Trend']
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=[f"Day {i+1}" for i in range(5)], y=trend_data, mode='lines+markers', name='Confidence'))
    fig_trend.update_layout(title=f"Confidence Trend (Last 5 Days)", yaxis_title="ML Score", yaxis_range=[0, 1])
    st.plotly_chart(fig_trend, width="stretch")
    
    st.markdown("---")
    st.subheader("NVIDIA AI Analysis")
    
    # Derive data source intelligently rather than hardcoding
    data_source_map = {
        "Trade Break": "Broker vs Internal",
        "Custodian Break": "Custodian vs Internal",
        "Cash Break": "Bank vs Internal",
        "FX Break": "Market Data Provider",
        "Currency Break": "Market Data Provider",
        "Equity Break": "Pricing Vendor",
    }
    
    exception_inputs = {
        "Exception Type": exc_data['Break Type'],
        "Severity": "Critical" if exc_data['Status'] == 'Escalated' else ("High" if abs(exc_data['Net Discrepancy (USD)']) > 25000 else "Medium"),
        "NAV Impact (USD)": exc_data['Net Discrepancy (USD)'],
        "Fund / Strategy": exc_data['Fund Name'],
        "Data Source": data_source_map.get(exc_data['Break Type'], "External Vendor vs Internal"),
        "Confidence Score": exc_data['Confidence Score'],
        "Past Resolution Patterns": exc_data['Recommendation'],
        "Compliance Flags": exc_data['Status'] == 'Escalated' and exc_data['Age of Escalation (Days)'] > 7,
        "Time-Sensitivity": f"{exc_data['Age of Escalation (Days)']} Days Old"
    }

    engine = RoutingDecisionEngine(strategy="Hybrid")
    
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        st.write("**Routing & Impact Analysis (Llama 3.3 70B):**")
        with st.spinner("Analyzing routing and impact..."):
            routing_analysis = engine.determine_route(exception_inputs)
            st.info(routing_analysis)
            
    with col_ai2:
        st.write("**Root Cause Analysis (Nemotron Reasoning):**")
        
        import yfinance as yf
        
        mcp_market_price = "N/A"
        if "Tickers" in exc_data and isinstance(exc_data["Tickers"], list) and len(exc_data["Tickers"]) > 0:
            first_ticker = exc_data["Tickers"][0]
            with st.spinner(f"MCP Server fetching live market data for {first_ticker}..."):
                try:
                    stock = yf.Ticker(first_ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    mcp_market_price = f"${current_price:,.2f}"
                    logger.info(f"MCP Server (yfinance) fetched live price for {first_ticker}: {mcp_market_price}")
                except Exception as e:
                    mcp_market_price = "Unavailable (MCP Timeout)"
                    logger.error(f"MCP Server (yfinance) failed to fetch price for {first_ticker}: {e}")
                    
        rca_prompt = f"""
Perform a detailed Root Cause Analysis on the following NAV exception:
- **Break Type**: {exc_data['Break Type']}
- **Fund Name**: {exc_data['Fund Name']}
- **Asset Ticker(s)**: {", ".join(exc_data.get('Tickers', [])) if isinstance(exc_data.get('Tickers'), list) else 'Unknown'}
- **Quantity of Shares**: {exc_data.get('Quantity', 0):,.0f}
- **Internal Price (Per Share)**: ${exc_data.get('Internal Price (USD)', 0):,.2f}
- **Internal Book Value (Total Position)**: ${exc_data.get('Internal Value (USD)', 0):,.2f}
- **Custodian External Value (Total Position)**: ${exc_data.get('External Value (USD)', 0):,.2f}
- **MCP Live Market Price (Primary Asset)**: {mcp_market_price}
- **Net Discrepancy (Total USD)**: ${exc_data['Net Discrepancy (USD)']:,.2f}
- **Potential Reason**: {exc_data['Possible Reason']}
- **Age**: {exc_data['Age of Escalation (Days)']} days

Please format your response clearly into the following sections:
1. **Executive Summary**: Brief overview of the break.
2. **Pricing Discrepancy Analysis**: Analyze the difference between the Internal Book Value, the Custodian's External Value, and how the current MCP Live Market Price (Per Share) might explain the discrepancy (e.g. stale pricing or incorrect share quantity).
3. **Operational Failure Mechanics**: Explain step-by-step how this type of break occurs.
4. **Downstream NAV Impact**: How this specific discrepancy affects the final NAV calculation.
5. **Recommended Resolution Workflow**: A clear action plan to resolve it.
"""
        with st.expander("View RCA Prompt Context (with Live MCP Data)"):
            st.code(rca_prompt, language="markdown")
            
        with st.spinner("Performing deep Root Cause Analysis using Internal & Market Data..."):
            rca_result = engine.ai_layer.reasoning_nim.invoke(rca_prompt)
            st.markdown(f"> {rca_result}")
    
    
    st.stop() # Do not render the rest of the page

# Sidebar Filters
st.sidebar.header("Filters")
selected_fund = st.sidebar.selectbox("Select Fund", ["All"] + list(df['Fund Name'].unique()))
selected_status = st.sidebar.selectbox("Status", ["All"] + list(df['Status'].unique()))

filtered_df = df.copy()
if selected_fund != "All":
    filtered_df = filtered_df[filtered_df['Fund Name'] == selected_fund]
if selected_status != "All":
    filtered_df = filtered_df[filtered_df['Status'] == selected_status]



# KPIs
st.subheader("NAV Funds Details")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Exceptions", len(filtered_df))
col2.metric("Open / Escalated", len(filtered_df[filtered_df['Status'].isin(['Open', 'Escalated'])]))
col3.metric("Total Discrepancy Impact", f"${filtered_df['Net Discrepancy (USD)'].sum():,.2f}")
col4.metric("Avg Escalation Age", f"{filtered_df['Age of Escalation (Days)'].mean():.1f} Days")

st.markdown("---")

# Visualizations
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Exceptions by Break Type")
    fig_pie = px.pie(filtered_df, names='Break Type', title="Break Types Distribution", hole=0.4)
    st.plotly_chart(fig_pie, width="stretch")

with col_chart2:
    st.subheader("Net Discrepancy vs Confidence")
    fig_scatter = px.scatter(
        filtered_df, 
        x='Net Discrepancy (USD)', 
        y='Confidence Score', 
        color='Status',
        hover_data=['Exception ID', 'Fund Name'],
        title="Impact vs ML Confidence"
    )
    st.plotly_chart(fig_scatter, width="stretch")

# Data Table
st.subheader("Exception Details (Generated by Reconciliation Skill)")

# Format dataframe for display
display_df = filtered_df[['Exception ID', 'Fund Name', 'Tickers', 'Break Type', 'Net Discrepancy (USD)', 'Confidence Score', 'Age of Escalation (Days)', 'Status', 'Recommendation']].copy()

# Join tickers for display
display_df['Tickers'] = display_df['Tickers'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

# Make the Exception ID itself a URL
display_df["Exception ID"] = "/?exception_id=" + display_df["Exception ID"]

# Render table
st.dataframe(
    display_df,
    column_config={
        "Exception ID": st.column_config.LinkColumn("Exception ID", display_text=r"exception_id=(.*)"),
        "Net Discrepancy (USD)": st.column_config.NumberColumn(format="$%d"),
        "Confidence Score": st.column_config.NumberColumn(format="%.2f")
    },
    hide_index=True,
    width="stretch"
)
