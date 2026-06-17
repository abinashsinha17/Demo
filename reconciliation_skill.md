# Reconciliation Skill

This document defines the automated logic for the **Reconciliation Skill** agent within the NAV Exception Management ecosystem.

## Overview
The Reconciliation Skill is responsible for evaluating breaks between internal systems and external data providers (like custodians). It examines parameters such as:
- Break Type (e.g., Cash Break, Position Break)
- Variance (The absolute dollar difference)
- Confidence Score (Machine learning probability)
- Age of Escalation

## Skill Logic Implementation

The following Python code defines the `ReconciliationSkill` class used by the system to process exceptions.

```python
import pandas as pd
import logging
import uuid

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
        
        # Exception ID is generated at runtime when the exception is raised/processed
        if 'Exception ID' not in processed_df.columns:
            processed_df['Exception ID'] = [f"EXC-{uuid.uuid4().hex[:8].upper()}" for _ in range(len(processed_df))]
            
        logger.info(f"Processing {len(processed_df)} exceptions...")
        
        # Rule 1: High variance cash breaks get escalated immediately
        mask_high_variance = (processed_df['Break Type'] == 'Cash Break') & (processed_df['Variance (USD)'].abs() > 25000)
        processed_df.loc[mask_high_variance, 'Status'] = 'Escalated'
        
        # Rule 2: Small position breaks with high confidence might be auto-resolved
        mask_auto_resolve = (processed_df['Break Type'] == 'Position Break') & (processed_df['Variance (USD)'].abs() < 500) & (processed_df['Confidence Score'] > 0.90)
        processed_df.loc[mask_auto_resolve, 'Status'] = 'Resolved'
        
        # Rule 3: Old exceptions drop in confidence
        mask_old = processed_df['Age of Escalation (Days)'] > 10
        processed_df.loc[mask_old, 'Confidence Score'] = processed_df.loc[mask_old, 'Confidence Score'] * 0.9
        
        # Add a resolution recommendation column
        recommendations = []
        for index, row in processed_df.iterrows():
            if row['Status'] == 'Resolved':
                recommendations.append("Auto-booked adjustment")
            elif row['Status'] == 'Escalated':
                recommendations.append("Requires manual review by Fund Controller")
            else:
                recommendations.append("Investigate missing source file")
        
        processed_df['Recommendation'] = recommendations
        
        logger.info(f"Successfully processed exceptions. Auto-resolved: {len(processed_df[processed_df['Status'] == 'Resolved'])}, Escalated: {len(processed_df[processed_df['Status'] == 'Escalated'])}")
        
        return processed_df
```
