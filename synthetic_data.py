import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import uuid
import os 

logger = logging.getLogger("SyntheticData")

def generate_synthetic_data(num_records=500):
    logger.info(f"Generating {num_records} synthetic exception records...")
    np.random.seed(42)
    
    fund_ticker_map = {
        "Alpha Equity Fund": ["AAPL", "MSFT", "GOOGL"],
        "Global Macro Fund": ["NVDA", "TSLA", "META", "AMZN"],
        "Yield Plus Fund": ["JPM", "BAC", "WFC", "GS"],
        "Emerging Markets Fund": ["BABA", "JD", "BIDU", "TCEHY"]
    }
    funds = list(fund_ticker_map.keys())
    break_types = [
        "Cash Break", "Position Break", "Trade Break", 
        "FX Break", "Custodian Break", "Currency Break", 
        "Equity Break", "Advt Break"
    ]
    
    data = []
    for i in range(num_records):
        fund = np.random.choice(funds)
        num_tickers = np.random.randint(1, 4)
        num_tickers = min(num_tickers, len(fund_ticker_map[fund]))
        selected_tickers = list(np.random.choice(fund_ticker_map[fund], size=num_tickers, replace=False))
        break_type = np.random.choice(break_types)
        
        quantity = np.random.randint(100, 10000)
        internal_price = round(np.random.uniform(50.0, 500.0), 2)
        random_price_offset = round(np.random.uniform(-5.0, 5.0), 2)
        external_price = round(internal_price + random_price_offset, 2)
        
        internal_value = round(float(quantity * internal_price), 2)
        
        record = {
            "Exception ID": f"EXC-{uuid.uuid4().hex[:8].upper()}",
            "Fund Name": fund,
            "Tickers": selected_tickers,
            "Break Type": break_type,
            "Quantity": quantity,
            "Internal Price (USD)": internal_price,
            "Internal Value (USD)": internal_value,
            "Status": "Pending",
            "Detected At": (datetime.now() - timedelta(days=np.random.randint(1, 15))).strftime("%Y-%m-%d %H:%M:%S")
        }
        data.append(record)
        
    logger.info("Synthetic data generation complete.")
    return pd.DataFrame(data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Increased data volume for better ML training
    df = generate_synthetic_data(1000)
    
    data_dir = "e:\\Project\\data"
    os.makedirs(data_dir, exist_ok=True)
    
    for break_type in df['Break Type'].unique():
        file_name = break_type.lower().replace(" ", "_") + "s.json"
        file_path = os.path.join(data_dir, file_name)
        subset = df[df['Break Type'] == break_type]
        subset.to_json(file_path, orient="records", indent=4)
        logger.info(f"Saved {len(subset)} records to {file_path}")
        
    old_file = os.path.join(data_dir, "exceptions.json")
    if os.path.exists(old_file):
        os.remove(old_file)
