import pandas as pd
import numpy as np
import glob
import os
import pickle
import json
import logging
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NAV_XGB")

def engineer_features(df):
    logger.info("Performing Base Feature Engineering...")
    
    df = df.copy()

    # 1. Re-calculate removed numerical fields via data comparison
    df['External Value (USD)'] = round(df['Quantity'] * df['External Price (USD)'], 2)
    df['Net Discrepancy (USD)'] = round(df['Internal Value (USD)'] - df['External Value (USD)'], 2)
    
    # Calculate age
    from datetime import datetime
    df['Detected At'] = pd.to_datetime(df['Detected At'])
    df['Age of Escalation (Days)'] = (datetime.now() - df['Detected At']).dt.days
    
    # 2. Auto-label historical data for XGBoost (since Recommendation was removed from raw JSON)
    def auto_label(row):
        abs_disc = abs(row['Net Discrepancy (USD)'])
        age = row['Age of Escalation (Days)']
        bt = row['Break Type']
        
        if abs_disc > 50000 or age > 15:
            status = 'Escalated'
        elif abs_disc < 5000 and age <= 3 and bt in ['Cash Break', 'Position Break']:
            status = 'Resolved'
        elif row['Fund Name'] in ['Alpha Equity Fund', 'Global Macro Fund'] and abs_disc > 15000:
            status = 'Escalated'
        elif abs_disc > 10000 and bt == 'Trade Break':
            status = 'Under Investigation'
        elif age <= 5 and abs_disc < 10000:
            status = 'Resolved'
        else:
            status = 'Under Investigation'
            
        if np.random.rand() < 0.05:
            status = np.random.choice(['Escalated', 'Resolved', 'Under Investigation'])
            
        if status == 'Escalated':
            return "Requires manual review by Fund Controller"
        elif status == 'Resolved':
            return "Auto-booked adjustment"
        else:
            return "Investigate missing source file"

    df['Recommendation'] = df.apply(auto_label, axis=1)

    required = [
        'Quantity',
        'Internal Value (USD)',
        'Age of Escalation (Days)',
        'Net Discrepancy (USD)',
        'Break Type',
        'Fund Name',
        'Recommendation'
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df.fillna({
        'Quantity': 0,
        'Internal Value (USD)': 0,
        'Age of Escalation (Days)': 0,
        'Net Discrepancy (USD)': 0,
        'Break Type': 'Unknown',
        'Fund Name': 'Unknown',
        'Recommendation': 'Unknown'
    }, inplace=True)

    df['Abs Discrepancy (USD)'] = df['Net Discrepancy (USD)'].abs()

    df['Age of Escalation (Days)'] = pd.to_numeric(
        df['Age of Escalation (Days)'],
        errors='coerce'
    ).fillna(0)

    df['Risk Index'] = (
        df['Age of Escalation (Days)']
        * np.log1p(df['Abs Discrepancy (USD)'])
    )

    return df


def train_model():

    files = glob.glob(r"e:\\Project\\data\\*.json")

    if not files:
        raise FileNotFoundError("No training files found")

    df = pd.concat(
        [pd.read_json(f) for f in files],
        ignore_index=True
    )

    df = engineer_features(df)

    counts = df['Recommendation'].value_counts()

    rare_classes = counts[counts < 5].index

    df['Recommendation'] = df['Recommendation'].replace(
        rare_classes,
        'Other'
    )

    features = [
        'Abs Discrepancy (USD)',
        'Age of Escalation (Days)',
        'Risk Index',
        'Break Type',
        'Fund Name'
    ]

    X = df[features]
    y_raw = df['Recommendation']

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw,
        stratify=y_raw,
        test_size=0.2,
        random_state=42
    )

    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    cat_cols = ['Break Type', 'Fund Name']

    X_train[cat_cols] = ordinal.fit_transform(
        X_train[cat_cols]
    )

    X_test[cat_cols] = ordinal.transform(
        X_test[cat_cols]
    )

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),

        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,

        subsample=0.8,
        colsample_bytree=0.8,

        tree_method="hist",

        eval_metric="mlogloss",

        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)

    metrics = {

        "Accuracy":
            accuracy_score(y_test, preds),

        "Precision":
            precision_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                preds,
                average="weighted",
                zero_division=0
            )
    }

    try:
        metrics["AUC"] = roc_auc_score(
            y_test,
            probas,
            multi_class="ovr",
            average="weighted"
        )
    except:
        metrics["AUC"] = 0

    try:
        metrics["LogLoss"] = log_loss(
            y_test,
            probas
        )
    except:
        metrics["LogLoss"] = 0

    logger.info(metrics)

    model_package = {
        "model": model,
        "ordinal_encoder": ordinal,
        "label_encoder": label_encoder,
        "features": features,
        "metrics": metrics
    }

    os.makedirs(r"e:\\Project\\models", exist_ok=True)

    with open(
        r"e:\\Project\\models\\nav_xgb.pkl",
        "wb"
    ) as f:
        pickle.dump(model_package, f)

    with open(
        r"e:\\Project\\models\\nav_metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    logger.info("Training completed successfully")


if __name__ == "__main__":
    train_model()