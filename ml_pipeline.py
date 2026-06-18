import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import pickle
import glob
import logging
try:
    import shap
except ImportError:
    shap = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NAV_XGB")

def engineer_base_features(df):
    logger.info("Performing Vectorized Feature Engineering...")
    
    df = df.copy()

    from datetime import datetime
    df['Detected At'] = pd.to_datetime(df['Detected At'], errors="coerce")
    
    age_days = (datetime.now() - df['Detected At']).dt.days
    df['Age of Escalation (Days)'] = pd.to_numeric(age_days, errors='coerce').fillna(0)
    
    df["Abs Discrepancy (USD)"] = df["Internal Value (USD)"].abs()
    
    # Historical Label Simulation (with noise to avoid strict rule memorization)
    np.random.seed(42)
    
    conditions = [
        (df["Abs Discrepancy (USD)"] > 50000) | (df["Age of Escalation (Days)"] > 15),
        (df["Abs Discrepancy (USD)"] < 5000) & (df["Age of Escalation (Days)"] <= 3) & (df["Break Type"].isin(["Cash Break", "Position Break"])),
        (df["Fund Name"].isin(["Alpha Equity Fund", "Global Macro Fund"])) & (df["Abs Discrepancy (USD)"] > 15000),
        (df["Abs Discrepancy (USD)"] > 10000) & (df["Break Type"] == "Trade Break")
    ]
    
    severity_choices = ["High", "Low", "High", "Medium"]
    status_choices = ["Escalated", "Resolved", "Escalated", "Under Investigation"]
    
    df["Severity"] = np.select(conditions, severity_choices, default="Medium")
    df["Status"] = np.select(conditions, status_choices, default="Under Investigation")
    
    base_priorities = [9.0, 2.5, 8.0, 5.5]
    base_pri = np.select(conditions, base_priorities, default=4.0)
    df["Priority Score"] = np.clip(base_pri + np.random.normal(0, 1.0, len(df)), 1.0, 10.0)

    noise_mask = np.random.rand(len(df)) < 0.15
    if noise_mask.sum() > 0:
        df.loc[noise_mask, "Status"] = np.random.choice(['Escalated', 'Resolved', 'Under Investigation'], size=noise_mask.sum())
        df.loc[noise_mask, "Severity"] = np.random.choice(['High', 'Medium', 'Low'], size=noise_mask.sum())
        
    rec_conditions = [df["Status"] == "Escalated", df["Status"] == "Resolved"]
    rec_choices = ["Requires manual review by Fund Controller", "Auto-booked adjustment"]
    df["Recommendation"] = np.select(rec_conditions, rec_choices, default="Investigate missing source file")

    required = ['Quantity', 'Internal Value (USD)', 'Age of Escalation (Days)', 'Break Type', 'Fund Name', 'Recommendation', 'Severity', 'Priority Score']
    df.fillna({
        'Quantity': 0, 'Internal Value (USD)': 0, 'Age of Escalation (Days)': 0,
        'Break Type': 'Unknown', 'Fund Name': 'Unknown', 'Recommendation': 'Unknown',
        'Severity': 'Medium', 'Priority Score': 5.0
    }, inplace=True)

    df["Log_Discrepancy"] = np.log1p(df["Abs Discrepancy (USD)"])
    df["Risk Index"] = df["Age of Escalation (Days)"] * df["Log_Discrepancy"]
    df["High_Value_Flag"] = (df["Abs Discrepancy (USD)"] > 50000).astype(int)
    df["Old_Exception_Flag"] = (df["Age of Escalation (Days)"] > 15).astype(int)

    return df

def train_model():
    logger.info("Scanning for JSON data files to begin training...")
    files = glob.glob(r"e:\\Project\\data\\*.json")

    if not files:
        raise FileNotFoundError("No training files found")

    df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)

    df = engineer_base_features(df)
    
    logger.info("Class Distributions before split:")
    logger.info(f"\n{df['Recommendation'].value_counts()}")
    logger.info(f"\n{df['Severity'].value_counts()}")

    counts = df['Recommendation'].value_counts()
    rare_classes = counts[counts < 5].index
    df['Recommendation'] = df['Recommendation'].replace(rare_classes, 'Other')

    features = [
        'Abs Discrepancy (USD)', 'Age of Escalation (Days)', 'Risk Index',
        'Log_Discrepancy', 'Fund_Risk', 'Break_Frequency', 'High_Value_Flag', 'Old_Exception_Flag',
        'Break Type', 'Fund Name'
    ]

    le_rec = LabelEncoder()
    y_rec = le_rec.fit_transform(df['Recommendation'])
    
    le_sev = LabelEncoder()
    y_sev = le_sev.fit_transform(df['Severity'])
    
    y_pri = df['Priority Score'].values
    
    # Proper stratified split on indices to avoid leakage
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y_rec, random_state=42
    )
    
    fund_risk_map = df.iloc[train_idx].groupby("Fund Name")["Abs Discrepancy (USD)"].mean().to_dict()
    break_freq_map = df.iloc[train_idx].groupby("Break Type")["Break Type"].count().to_dict()
    
    df["Fund_Risk"] = df["Fund Name"].map(fund_risk_map).fillna(df.iloc[train_idx]["Abs Discrepancy (USD)"].mean())
    df["Break_Frequency"] = df["Break Type"].map(break_freq_map).fillna(1)
    
    X = df[features].copy()
    
    for col in ["Break Type", "Fund Name"]:
        X[col] = X[col].astype("category")
        
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_rec_train, y_rec_test = y_rec[train_idx], y_rec[test_idx]
    y_sev_train, y_sev_test = y_sev[train_idx], y_sev[test_idx]
    y_pri_train, y_pri_test = y_pri[train_idx], y_pri[test_idx]

    def build_classifier(num_class):
        return xgb.XGBClassifier(
            objective="multi:softprob", num_class=num_class,
            enable_categorical=True, tree_method="hist",
            max_depth=8, learning_rate=0.03, n_estimators=1000,
            min_child_weight=5, gamma=0.1, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=1.5,
            eval_metric="mlogloss", random_state=42, early_stopping_rounds=30
        )

    logger.info("Training XGBoost Recommendation Classifier...")
    model_rec = build_classifier(len(le_rec.classes_))
    model_rec.fit(X_train, y_rec_train, eval_set=[(X_test, y_rec_test)], sample_weight=compute_sample_weight("balanced", y_rec_train), verbose=False)

    logger.info("Training XGBoost Severity Classifier...")
    model_sev = build_classifier(len(le_sev.classes_))
    model_sev.fit(X_train, y_sev_train, eval_set=[(X_test, y_sev_test)], sample_weight=compute_sample_weight("balanced", y_sev_train), verbose=False)
    
    logger.info("Training XGBoost Priority Regressor...")
    model_pri = xgb.XGBRegressor(
        objective="reg:squarederror", enable_categorical=True, tree_method="hist",
        max_depth=8, learning_rate=0.03, n_estimators=1000,
        min_child_weight=5, gamma=0.1, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.5, reg_lambda=1.5, random_state=42, early_stopping_rounds=30
    )
    model_pri.fit(X_train, y_pri_train, eval_set=[(X_test, y_pri_test)], verbose=False)

    def evaluate_classification(model, X_t, y_t):
        probas = model.predict_proba(X_t)
        preds = np.argmax(probas, axis=1)
        m = {
            "Accuracy": accuracy_score(y_t, preds),
            "Precision": precision_score(y_t, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_t, preds, average="weighted", zero_division=0),
            "F1": f1_score(y_t, preds, average="weighted", zero_division=0),
        }
        try:
            m["AUC"] = roc_auc_score(y_t, probas, multi_class="ovr", average="weighted")
        except ValueError:
            pass
        m["LogLoss"] = log_loss(y_t, probas)
        return m
        
    def evaluate_regression(model, X_t, y_t):
        preds = model.predict(X_t)
        return {
            "MAE": mean_absolute_error(y_t, preds),
            "RMSE": np.sqrt(mean_squared_error(y_t, preds)),
            "R2": r2_score(y_t, preds)
        }

    logger.info("Evaluating models...")
    metrics = {
        "Recommendation": evaluate_classification(model_rec, X_test, y_rec_test),
        "Severity": evaluate_classification(model_sev, X_test, y_sev_test),
        "Priority": evaluate_regression(model_pri, X_test, y_pri_test)
    }
    
    logger.info(f"Model Metrics: {json.dumps(metrics, indent=2)}")

    booster = model_rec.get_booster()
    scores = booster.get_score(importance_type="gain")
    importance = pd.DataFrame([{"feature": k, "score": v} for k, v in scores.items()])
    importance = importance.sort_values("score", ascending=False)
    
    model_dir = r"e:\\Project\\models"
    os.makedirs(model_dir, exist_ok=True)
    importance.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)

    logger.info("Serializing individual model artifacts...")
    model_rec.save_model(os.path.join(model_dir, "recommendation.json"))
    model_sev.save_model(os.path.join(model_dir, "severity.json"))
    model_pri.save_model(os.path.join(model_dir, "priority.json"))
    
    metadata = {
        "le_recommendation": le_rec,
        "le_severity": le_sev,
        "features": features,
        "metrics": metrics,
        "fund_risk_map": fund_risk_map,
        "break_freq_map": break_freq_map,
        "mean_fund_risk": df.iloc[train_idx]["Abs Discrepancy (USD)"].mean()
    }
    with open(os.path.join(model_dir, "nav_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    with open(os.path.join(model_dir, "nav_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    try:
        if shap is not None:
            explainer = shap.TreeExplainer(model_rec)
            shap_values = explainer.shap_values(X_test)
            logger.info("SHAP values calculated successfully.")
        else:
            logger.warning("SHAP module not installed. Skipping local SHAP explanations. (Run 'pip install shap' to enable)")
    except Exception as e:
        logger.warning(f"Could not calculate SHAP values: {e}")

    logger.info("Training completed successfully")

if __name__ == "__main__":
    train_model()