import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("SyntheticData")

def generate_synthetic_data(num_records=50):
    logger.info(f"Generating {num_records} synthetic exception records...")
    np.random.seed(42)
    
    funds = ["Alpha Equity Fund", "Global Macro Fund", "Yield Plus Fund", "Emerging Markets Fund"]
    break_types = ["Cash Break", "Position Break", "Trade Break", "FX Break", "Custodian Break"]
    statuses = ["Open", "Under Investigation", "Escalated", "Resolved"]
    
    data = []
    for i in range(num_records):
        fund = np.random.choice(funds)
        break_type = np.random.choice(break_types)
        variance = round(np.random.uniform(-50000, 50000), 2)
        confidence_score = round(np.random.uniform(0.6, 0.99), 2)
        age_days = np.random.randint(1, 15)
        status = np.random.choice(statuses, p=[0.4, 0.3, 0.1, 0.2])
        
        # Trend over last 5 days
        trend = [round(min(1.0, max(0.4, confidence_score + np.random.uniform(-0.1, 0.1))), 2) for _ in range(5)]
        
        # Generate mock root causes
        if break_type == "Cash Break":
            root_cause = "Missing wire transfer confirmation"
            possible_reason = "Counterparty delay in settling cash or mismatched settlement dates."
        elif break_type == "Position Break":
            root_cause = "Corporate action not applied"
            possible_reason = "Stock split or dividend wasn't processed in internal accounting system."
        elif break_type == "Trade Break":
            root_cause = "Price mismatch on execution"
            possible_reason = "Broker execution price differs from internal fill due to slippage."
        elif break_type == "FX Break":
            root_cause = "Stale FX rate used"
            possible_reason = "System failed to fetch the latest EOD spot rate from data provider."
        else:
            root_cause = "Data file formatting error"
            possible_reason = "Custodian sent the feed with an unexpected delimiter or missing column."
            
        record = {
            "Fund Name": fund,
            "Break Type": break_type,
            "Variance (USD)": variance,
            "Confidence Score": confidence_score,
            "Confidence Trend": trend,
            "Age of Escalation (Days)": age_days,
            "Status": status,
            "Detected At": (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d %H:%M:%S"),
            "Root Cause": root_cause,
            "Possible Reason": possible_reason
        }
        data.append(record)
        
    logger.info("Synthetic data generation complete.")
    return pd.DataFrame(data)

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    df = generate_synthetic_data(100)
    os.makedirs("data", exist_ok=True)
    df.to_json("data/exceptions.json", orient="records", indent=4)
    logger.info("Saved generated data to data/exceptions.json")
