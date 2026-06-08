import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

def run_shap_explainability(file_path='creditcard.csv'):
    # Prep data
    df = pd.read_csv(file_path).dropna()
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    # Retrain the Best LightGBM model
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    
    best_params = {
        'n_estimators': 68,
        'learning_rate': 0.0111,
        'num_leaves': 36,
        'max_depth': 7,
        'scale_pos_weight': neg_count / pos_count,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    print("Training LightGBM model for SHAP Explainability...")
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_scaled, y_train)

    print("\n--- Generating SHAP Explanations ---")
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # We sample up to 1000 rows from the test set to prevent memory issues
    sample_size = min(1000, len(X_test_scaled))
    X_test_sample = X_test_scaled.sample(n=sample_size, random_state=42)
    
    print("Calculating SHAP values (this might take a few seconds)...")
    shap_values = explainer.shap_values(X_test_sample)
    
    # 1. Global Explainability
    print("\nPlotting Global Feature Importance (Summary Plot)...")
    # For LightGBM, shap_values is sometimes a list [Class 0, Class 1]. We want Class 1 (Fraud)
    shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Summary Plot - Global Importance")
    shap.summary_plot(shap_vals_to_plot, X_test_sample, show=True)
    
    # 2. Local Explainability
    fraud_indices = y_test[y_test == 1].index
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        iloc_idx = X_test_scaled.index.get_loc(fraud_idx)
        
        print("\nPlotting Local Explanation (Waterfall Plot) for a specific Fraud Transaction...")
        # Explainer object callable for Waterfall plot
        explanation = explainer(X_test_scaled.iloc[[iloc_idx]])
        
        plt.figure()
        if len(explanation.shape) == 3:
            shap.plots.waterfall(explanation[0, :, 1])
        else:
            shap.plots.waterfall(explanation[0])
    else:
        print("No fraud cases found in the test set to explain.")

if __name__ == "__main__":
    run_shap_explainability()
