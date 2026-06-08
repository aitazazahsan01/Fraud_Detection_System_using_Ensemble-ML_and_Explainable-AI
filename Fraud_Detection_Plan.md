# Fraud Detection System - Low-Resource Project Plan

This document outlines the strategy for building an end-to-end Fraud Detection System (Ensemble ML + Explainable AI) optimized for free Google Colab instances and low-resource local machines.

## 🛠️ Adaptations for Low-Resource Environments

1. **Dataset Selection**: We will **exclusively use the Kaggle Credit Card Fraud dataset**. It is ~150MB (284k rows, 30 features), which easily fits into Colab's free RAM (usually 12-16GB). We will skip the IEEE-CIS dataset (590k rows, 400+ features) as it requires heavy memory for data manipulation and training.
2. **Algorithm Choice**: We will prioritize **LightGBM** over XGBoost. LightGBM is specifically designed for high speed and low memory usage through histogram-based algorithms, making it perfect for constrained environments.
3. **Hyperparameter Tuning**: We will cap Optuna hyperparameter searches to a small number of trials (e.g., 20-30). This is enough to get a strong model without running Colab out of execution time.
4. **SHAP Explainability**: Calculating SHAP values for the entire dataset is extremely computationally expensive. We will compute SHAP values only on a **representative sample** of the test data (e.g., 1,000 to 2,000 rows) to prevent RAM crashes.
5. **Decoupled Architecture**: 
   - **Training (Heavy lifting)**: Done entirely on Google Colab.
   - **Deployment (Lightweight)**: Done locally on your machine using FastAPI. Serving a trained LightGBM model requires almost zero CPU and very little RAM (<100MB).

---

## 🧰 Tech Stack & Tools

*   **Environment**: Google Colab (Training) & Local Machine (Serving)
*   **Data Processing**: `pandas`, `numpy`
*   **Machine Learning**: `scikit-learn`, `imbalanced-learn` (for SMOTE), `lightgbm`, `xgboost`
*   **Hyperparameter Tuning**: `optuna`
*   **Explainable AI**: `shap`
*   **Experiment Tracking**: `mlflow` (using Colab's local filesystem or sqlite backend)
*   **Deployment**: `fastapi`, `uvicorn` (Local)

---

## 📋 Step-by-Step Execution Plan

### Phase 1: Environment Setup & EDA
*   **Action**: Start a Google Colab notebook. Download the Kaggle Credit Card Fraud dataset.
*   **Tasks**: 
    *   Perform Exploratory Data Analysis (EDA) on the 284,807 transactions.
    *   Plot the extreme class distribution (0.17% fraud).
    *   Analyze feature distributions and check for missing values (the Kaggle dataset is pre-cleaned with PCA features, so this will be quick).
    *   *Output*: Understand the precision-recall trade-off and why standard accuracy is misleading.

### Phase 2: Baseline Models
*   **Action**: Train fast, simple models to set a benchmark.
*   **Tasks**:
    *   Split data into Train/Test sets with stratification.
    *   Train a Logistic Regression model.
    *   Train a small Random Forest (e.g., `n_estimators=50`, `max_depth=10` to save RAM).
    *   Evaluate using Precision, Recall, F1-Score, and PR-AUC.

### Phase 3: Handling Class Imbalance
*   **Action**: Experiment with data-level vs. algorithm-level imbalance handling.
*   **Tasks**:
    *   Apply SMOTE (Synthetic Minority Over-sampling Technique) to the training set only.
    *   *Resource Tip*: Avoid ADASYN as it can be slower; stick to standard SMOTE.
    *   Compare the performance of Baseline models with SMOTE vs. Class Weights (`class_weight='balanced'`).

### Phase 4: Gradient Boosting Ensemble (The Core)
*   **Action**: Build the primary predictive models.
*   **Tasks**:
    *   Train a LightGBM classifier. It handles imbalance natively using `scale_pos_weight` or `is_unbalance=True`.
    *   Set up an Optuna study with ~20 trials to find the best `learning_rate`, `num_leaves`, and `max_depth`.
    *   *Optional*: Train a small XGBoost model and use a simple soft-voting ensemble if RAM permits.

### Phase 5: SHAP Explainability
*   **Action**: Interpret the model decisions.
*   **Tasks**:
    *   Initialize `shap.TreeExplainer` on the best LightGBM model.
    *   Extract a sample of 1,000 rows from the test set.
    *   Generate a SHAP Summary Plot (global explainability) and a Waterfall plot for one specific fraudulent transaction (local explainability).

### Phase 6: MLflow Tracking & FastAPI Production
*   **Action**: Bridge the gap from notebook to production.
*   **Tasks in Colab**:
    *   Wrap your final training run in an `mlflow.start_run()` block.
    *   Log parameters and the final PR-AUC metric.
    *   Save the best model using `joblib` (e.g., `fraud_model.joblib`) and download it to your local machine.
*   **Tasks Locally**:
    *   Write a `main.py` using FastAPI.
    *   Load `fraud_model.joblib` into memory on startup.
    *   Create a POST endpoint `/predict` that accepts transaction features and returns a fraud probability < 50ms.
    *   Run locally using `uvicorn main:app --reload`.
