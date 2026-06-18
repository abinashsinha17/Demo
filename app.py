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
        st.metric("Internal Impact", f"${exc_data['Internal Value (USD)']:,.2f}")
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
    st.subheader("Machine Learning Explainability (SHAP)")
    
    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        st.write(f"**Exception ID**: {exc_data['Exception ID']}")
        st.write(f"**Predicted Severity**: {exc_data['Scenario']}")
        st.write(f"**Confidence**: {exc_data['Confidence Score']*100:.1f}%")
        
    with col_shap2:
        try:
            import shap
            import xgboost as xgb
            import pickle
            
            metadata_path = r"e:\\Project\\models\\nav_metadata.pkl"
            rec_path = r"e:\\Project\\models\\recommendation.json"
            
            if os.path.exists(metadata_path) and os.path.exists(rec_path):
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                features = metadata['features']
                
                # Prepare single row for SHAP inference
                shap_row = exc_data[features].to_frame().T
                for col in features:
                    if col in ['Break Type', 'Fund Name']:
                        shap_row[col] = shap_row[col].astype('category')
                    else:
                        shap_row[col] = pd.to_numeric(shap_row[col], errors='coerce')
                    
                model_rec = xgb.XGBClassifier()
                model_rec.load_model(rec_path)
                
                explainer = shap.TreeExplainer(model_rec)
                shap_vals = explainer.shap_values(shap_row)
                
                # Get prediction index to extract the correct SHAP matrix
                pred_idx = model_rec.predict(shap_row)[0]
                class_shap_vals = shap_vals[pred_idx][0]
                
                # Zip and sort top drivers
                driver_df = pd.DataFrame({
                    "Feature": features,
                    "Impact": class_shap_vals,
                    "Value": shap_row.iloc[0].values
                })
                driver_df['Abs_Impact'] = driver_df['Impact'].abs()
                top_drivers = driver_df.sort_values(by='Abs_Impact', ascending=False).head(3)
                
                st.write("**Top Drivers:**")
                for i, row in enumerate(top_drivers.itertuples(), 1):
                    # Format value cleanly if it's a float
                    val = f"${row.Value:,.2f}" if isinstance(row.Value, (int, float)) and 'USD' in row.Feature else row.Value
                    st.write(f"{i}. {row.Feature} = {val}")
                    
        except ImportError:
            st.warning("SHAP library is not installed or loading. Run 'pip install shap' to view live explainability.")
        except Exception as e:
            st.error(f"SHAP Explainer Error: {e}")
    
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
        "Severity": "Critical" if exc_data['Status'] == 'Escalated' else ("High" if abs(exc_data['Internal Value (USD)']) > 25000 else "Medium"),
        "NAV Impact (USD)": exc_data['Internal Value (USD)'],
        "Fund / Strategy": exc_data['Fund Name'],
        "Data Source": data_source_map.get(exc_data['Break Type'], "External Vendor vs Internal"),
        "Confidence Score": exc_data['Confidence Score'],
        "Past Resolution Patterns": exc_data['Recommendation'],
        "Compliance Flags": exc_data['Status'] == 'Escalated' and exc_data['Age of Escalation (Days)'] > 7,
        "Time-Sensitivity": f"{exc_data['Age of Escalation (Days)']} Days Old"
    }

    from agentic_workflow import build_agentic_workflow
    workflow = build_agentic_workflow()
    
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        st.write("**LangGraph Supervisor Orchestration:**")
        with st.spinner("Supervisor routing to specific skill nodes..."):
            initial_state = {
                "exception": {
                    "exception_id": exc_data['Exception ID'],
                    "break_type": exc_data['Break Type'],
                    "discrepancy_usd": exc_data['Internal Value (USD)'],
                    "status": "New"
                },
                "audit_trail": [],
                "evidence": [],
            }
            final_state = workflow.invoke(initial_state)
            
            st.success(f"**Supervisor Status:** {final_state.get('status')} via {final_state.get('next_skill', 'Agent')}")
            st.info(f"**Severity:** {final_state.get('severity')} | **Priority:** {final_state.get('priority_score')}")
            
            with st.expander("View LangGraph Audit Trail"):
                for log in final_state.get('audit_trail', []):
                    st.write(f"- {log}")
                    
            with st.expander("View Skill Evidence Collected"):
                for ev in final_state.get('evidence', []):
                    st.write(ev)
            
            st.write(f"**Triage Summary:** {final_state.get('triage_summary', 'N/A')}")
            
    with col_ai2:
        st.write("**Root Cause Analysis (Nemotron Reasoning):**")
        
        from nvidia_ai_layer import NVIDIAAILayer
        ai_layer = NVIDIAAILayer()
        
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
- **MCP Live Market Price (Primary Asset)**: {mcp_market_price}
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
            rca_result = ai_layer.reasoning_nim.invoke(rca_prompt)
            st.markdown(f"> {rca_result}")
    
    st.markdown("---")
    st.subheader("Human Approval & Continuous Learning")
    
    col_fb1, col_fb2 = st.columns([3, 1])
    with col_fb1:
        st.text_area("Resolution Feedback / Notes", value=exc_data.get('Recommendation', 'Reviewed and approved.'), height=100)
    
    with col_fb2:
        st.write("")
        st.write("")
        if st.button("Approve & Retrain XGBoost Model", type="primary"):
            with st.spinner("Updating Feedback Loop & Retraining XGBoost Model..."):
                try:
                    from ml_pipeline import train_model
                    train_model()
                    st.success("Success! XGBoost model retrained with new human feedback.")
                except Exception as e:
                    st.error(f"Failed to retrain model: {e}")
                    
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



# Format dataframe for display based on RVT table layout
display_df = filtered_df[['Exception ID', 'Fund Name', 'Tickers', 'Break Type', 'Internal Value (USD)', 'Variance (%)', 'Age of Escalation (Days)', 'Scenario', 'Confidence Score', 'Status']].copy()

# Join tickers for display
display_df['Tickers'] = display_df['Tickers'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
display_df['Age of Escalation (Days)'] = display_df['Age of Escalation (Days)'].apply(lambda x: f"{x}d")
display_df['Confidence Score'] = display_df['Confidence Score'] * 100

display_df = display_df.rename(columns={
    'Fund Name': 'Fund name',
    'Tickers': 'Entitys',
    'Break Type': 'Breaks',
    'Internal Value (USD)': 'Exposure',
    'Variance (%)': 'Variance',
    'Age of Escalation (Days)': 'Aging',
    'Scenario': 'Impact',
    'Confidence Score': 'Confidence'
})

display_df["Exception ID"] = "/?exception_id=" + display_df["Exception ID"]

def style_scenario(val):
    if 'Low' in str(val): return 'color: #198754; font-weight: bold'
    elif 'Medium' in str(val): return 'color: #fd7e14; font-weight: bold'
    elif 'High' in str(val): return 'color: #dc3545; font-weight: bold'
    return ''

def style_variance(val):
    if isinstance(val, (int, float)):
        if val <= 5.0: return 'color: #198754'
        elif val <= 20.0: return 'color: #fd7e14'
        else: return 'color: #dc3545'
    return ''

def style_fund(val):
    import hashlib
    h = int(hashlib.md5(str(val).encode('utf-8')).hexdigest(), 16)
    hue = h % 360
    return f'color: hsl({hue}, 70%, 40%); font-weight: bold'

try:
    styled_df = display_df.style.map(style_scenario, subset=['Impact'])\
                                .map(style_variance, subset=['Variance'])\
                                .map(style_fund, subset=['Fund name'])
except AttributeError:
    # Fallback for older pandas versions
    styled_df = display_df.style.applymap(style_scenario, subset=['Impact'])\
                                .applymap(style_variance, subset=['Variance'])\
                                .applymap(style_fund, subset=['Fund name'])

# Render table
st.dataframe(
    styled_df,
    column_config={
        "Exception ID": st.column_config.LinkColumn("ExceptionID", display_text=r"exception_id=(.*)"),
        "Exposure": st.column_config.NumberColumn(format="$%.2f"),
        "Variance": st.column_config.NumberColumn(format="%.1f%%"),
        "Confidence": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)
    },
    hide_index=True,
    width="stretch",
    height=600
)
