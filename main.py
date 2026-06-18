import pandas as pd
import re
import logging
from synthetic_data import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

from reconciliation_skill import ReconciliationSkill

if __name__ == "__main__":
    print("1. Loading synthetic records from data folder...")
    import glob
    import os
    try:
        data_dir = "e:\\Project\\data"
        json_files = glob.glob(os.path.join(data_dir, "*_breaks.json"))
        if json_files:
            df_raw = pd.concat([pd.read_json(f) for f in json_files], ignore_index=True)
            logger.info(f"Loaded synthetic data from {len(json_files)} files in data folder.")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        logger.error("Data files not found! Run synthetic_data.py first.")
        exit(1)
    print("\n--- Raw Data (First 3 rows) ---")
    print(df_raw[['Fund Name', 'Break Type', 'Net Discrepancy (USD)', 'Status']].head(3).to_string())
    
    print("\n2. Initializing Reconciliation Skill...")
    skill = ReconciliationSkill()
    print(f"Loaded Skill: {skill.name} - {skill.description}")
    
    print("\n3. Processing generated data...")
    df_processed = skill.process_exceptions(df_raw)
    
    print("\n--- Processed Data ---")
    columns_to_display = ['Exception ID', 'Break Type', 'Net Discrepancy (USD)', 'Status', 'Recommendation']
    print(df_processed[columns_to_display].to_string())
