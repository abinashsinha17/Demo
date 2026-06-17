import pandas as pd
import re
import logging
from synthetic_data import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def load_reconciliation_skill():
    # Read the logic from the markdown file
    with open("e:\\Project\\reconciliation_skill.md", "r") as f:
        md_content = f.read()

    # Extract the python code block
    code_match = re.search(r'```python(.*?)```', md_content, re.DOTALL)
    if code_match:
        skill_code = code_match.group(1).strip()
        # Execute the code block to make ReconciliationSkill available in the current namespace
        namespace = {}
        exec(skill_code, namespace)
        return namespace['ReconciliationSkill']
    else:
        raise ValueError("No Python code block found in reconciliation_skill.md")

if __name__ == "__main__":
    print("1. Loading synthetic records from data folder...")
    try:
        df_raw = pd.read_json("e:\\Project\\data\\exceptions.json")
        logger.info("Loaded synthetic data from data folder.")
    except FileNotFoundError:
        logger.error("Data file not found! Run synthetic_data.py first.")
        exit(1)
    print("\n--- Raw Data (First 3 rows) ---")
    print(df_raw[['Fund Name', 'Break Type', 'Variance (USD)', 'Status']].head(3).to_string())
    
    print("\n2. Loading Reconciliation Skill from markdown...")
    ReconciliationSkill = load_reconciliation_skill()
    skill = ReconciliationSkill()
    print(f"Loaded Skill: {skill.name} - {skill.description}")
    
    print("\n3. Processing generated data...")
    df_processed = skill.process_exceptions(df_raw)
    
    print("\n--- Processed Data ---")
    columns_to_display = ['Exception ID', 'Break Type', 'Variance (USD)', 'Status', 'Recommendation']
    print(df_processed[columns_to_display].to_string())
