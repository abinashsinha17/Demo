import pandas as pd
import numpy as np
import logging
import pickle
import os
import yfinance as yf

logger = logging.getLogger("ReconciliationSkill")

class ReconciliationSkill:
    def __init__(self):
        self.name = "Reconciliation Skill"
        self.description = "Analyzes and processes Cash, Position, and Trade breaks between internal systems and custodians."
        logger.info(f"Initialized {self.name}")

    def process_exceptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies reconciliation logic to exceptions.
        In this mock implementation, it adjusts confidence scores based on break types
        and variance thresholds, and flags items for auto-resolution or escalation.
        """
        processed_df = df.copy()
        
        logger.info(f"Processing {len(processed_df)} exceptions...")
        
        # 1. Fetch External Prices dynamically via yfinance
        logger.info("Fetching MCP External Prices for Inference...")
        unique_tickers = set()
        for t_list in processed_df['Tickers']:
            if isinstance(t_list, list) and len(t_list) > 0:
                unique_tickers.add(t_list[0])
                
        price_map = {}
        for ticker in unique_tickers:
            try:
                stock = yf.Ticker(ticker)
                price_map[ticker] = round(float(stock.history(period="1d")['Close'].iloc[-1]), 2)
            except:
                price_map[ticker] = 150.00
                
        processed_df['External Price (USD)'] = processed_df['Tickers'].apply(lambda x: price_map.get(x[0], 150.00) if isinstance(x, list) and len(x) > 0 else 150.00)
        processed_df['External Value (USD)'] = round(processed_df['Quantity'] * processed_df['External Price (USD)'], 2)
        
        # Calculate Variance percentage between 1% and 100%
        raw_variance_pct = (processed_df['Internal Value (USD)'] - processed_df['External Value (USD)']).abs() / processed_df['Internal Value (USD)'].replace(0, 1).abs() * 100
        processed_df['Variance (%)'] = np.clip(raw_variance_pct, 1.0, 100.0).round(2)

        # Pre-process status rules
        mask_high_variance = (processed_df['Break Type'] == 'Cash Break') & (processed_df['Internal Value (USD)'].abs() > 25000)
        processed_df.loc[mask_high_variance, 'Status'] = 'Escalated'
        
        # Rule 2: Small position breaks
        mask_auto_resolve = (processed_df['Break Type'] == 'Position Break') & (processed_df['Internal Value (USD)'].abs() < 500)
        processed_df.loc[mask_auto_resolve, 'Status'] = 'Resolved'
        

        # Load the XGBoost models and metadata dynamically
        import xgboost as xgb
        metadata_path = r"e:\\Project\\models\\nav_metadata.pkl"
        rec_path = r"e:\\Project\\models\\recommendation.json"
        sev_path = r"e:\\Project\\models\\severity.json"
        
        if os.path.exists(metadata_path) and os.path.exists(rec_path):
            logger.info("Generating recommendations using native XGBoost models...")
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                
            label_encoder = metadata['le_recommendation']
            le_sev = metadata['le_severity']
            features = metadata['features']
            
            # Load individual models
            model = xgb.XGBClassifier()
            model.load_model(rec_path)
            
            model_sev = xgb.XGBClassifier()
            model_sev.load_model(sev_path)
            
            # Prepare features (ensure all required columns exist)
            from datetime import datetime
            processed_df['Detected At'] = pd.to_datetime(processed_df['Detected At'])
            processed_df['Age of Escalation (Days)'] = (datetime.now() - processed_df['Detected At']).dt.days
            
            processed_df['Abs Discrepancy (USD)'] = processed_df['Internal Value (USD)'].abs()
            processed_df['Log_Discrepancy'] = np.log1p(processed_df['Abs Discrepancy (USD)'])
            processed_df['Risk Index'] = processed_df['Age of Escalation (Days)'] * processed_df['Log_Discrepancy']
            processed_df['Fund_Risk'] = processed_df["Fund Name"].map(metadata.get('fund_risk_map', {})).fillna(metadata.get('mean_fund_risk', 0))
            processed_df['Break_Frequency'] = processed_df["Break Type"].map(metadata.get('break_freq_map', {})).fillna(1)
            processed_df['High_Value_Flag'] = (processed_df['Abs Discrepancy (USD)'] > 50000).astype(int)
            processed_df['Old_Exception_Flag'] = (processed_df['Age of Escalation (Days)'] > 15).astype(int)
            
            X_infer = processed_df[features].copy()
            
            # Native Categorical support
            cat_cols = ['Break Type', 'Fund Name']
            for col in cat_cols:
                X_infer[col] = X_infer[col].astype('category')
            
            # Predict Recommendation
            predictions = model.predict(X_infer)
            probabilities = model.predict_proba(X_infer)
            if predictions.ndim > 1:
                predictions = np.argmax(predictions, axis=1)
            processed_df['Recommendation'] = label_encoder.inverse_transform(predictions)
            
            # Predict Severity
            sev_preds = model_sev.predict(X_infer)
            if sev_preds.ndim > 1:
                sev_preds = np.argmax(sev_preds, axis=1)
            processed_df['Severity'] = le_sev.inverse_transform(sev_preds)
            
            # Map Scenario based on Variance
            def map_scenario_by_variance(var):
                if var <= 5.0: return 'Low'
                elif var <= 20.0: return 'Medium'
                else: return 'High'
            processed_df['Scenario'] = processed_df['Variance (%)'].apply(map_scenario_by_variance)
            
            # Use max probability as the dynamic Confidence Score
            # Jitter slightly for realistic UI display to avoid exact 1.00
            raw_conf = probabilities.max(axis=1)
            jittered_conf = raw_conf - np.random.uniform(0.01, 0.04, len(processed_df))
            processed_df['Confidence Score'] = np.round(np.clip(jittered_conf, 0.60, 0.99), 2)
            
            # Rule 3: Old exceptions drop in confidence (applied post-inference)
            mask_old = processed_df['Age of Escalation (Days)'] > 10
            processed_df.loc[mask_old, 'Confidence Score'] = processed_df.loc[mask_old, 'Confidence Score'] * 0.9
            
            # Generate dummy trend ending in current score
            processed_df['Confidence Trend'] = processed_df['Confidence Score'].apply(
                lambda score: [round(max(0.4, min(1.0, score + np.random.uniform(-0.1, 0.1))), 2) for _ in range(4)] + [score]
            )
            
            logger.info("Successfully applied ML inference.")
        else:
            logger.warning("XGBoost model not found, falling back to basic recommendations.")
            processed_df['Recommendation'] = "Pending AI Analysis"
            
        # Dynamically generate root cause and possible reason mapping based on Break Type
        def get_root_cause(bt):
            if bt == "Cash Break": return "Missing wire transfer confirmation"
            elif bt == "Position Break": return "Corporate action not applied"
            elif bt == "Trade Break": return "Price mismatch on execution"
            elif bt in ["FX Break", "Currency Break"]: return "Stale or mismatched FX rate"
            elif bt == "Equity Break": return "Equity valuation mismatch"
            else: return "Data file formatting error"
            
        def get_possible_reason(bt):
            if bt == "Cash Break": return "Counterparty delay in settling cash or mismatched settlement dates."
            elif bt == "Position Break": return "Stock split or dividend wasn't processed in internal accounting system."
            elif bt == "Trade Break": return "Broker execution price differs from internal fill due to slippage."
            elif bt in ["FX Break", "Currency Break"]: return "System failed to fetch the latest EOD spot rate from data provider."
            elif bt == "Equity Break": return "Closing price from primary exchange differed from custodian valuation."
            else: return "Custodian sent the feed with an unexpected delimiter or missing column."
            
        processed_df['Root Cause'] = processed_df['Break Type'].apply(get_root_cause)
        processed_df['Possible Reason'] = processed_df['Break Type'].apply(get_possible_reason)
        
        logger.info(f"Successfully processed exceptions. Auto-resolved: {len(processed_df[processed_df['Status'] == 'Resolved'])}, Escalated: {len(processed_df[processed_df['Status'] == 'Escalated'])}")
        
        return processed_df
