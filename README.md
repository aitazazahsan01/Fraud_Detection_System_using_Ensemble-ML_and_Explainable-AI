# 🛡️ Fraud Detection System (Ensemble ML & Explainable AI)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136.3-009688.svg)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-orange.svg)

An end-to-end Machine Learning pipeline designed to detect credit card fraud in highly imbalanced datasets. This project demonstrates how banks and financial institutions tackle extreme class imbalance, optimize tree-based models, explain black-box AI decisions, and deploy models to a real-time production API.

## 🎯 Project Objectives
- **Master Extreme Class Imbalance**: Handle a dataset where frauds account for only **0.17%** of transactions.
- **Model Tuning**: Optimize Gradient Boosting models (LightGBM) using Optuna.
- **Explainability**: Comply with banking regulations by explaining *why* a transaction was declined using SHAP (Shapley Additive exPlanations).
- **Production Deployment**: Serve the trained model locally via a high-performance FastAPI endpoint with <50ms latency.

---

## 🛠️ Tech Stack
- **Data Processing**: `pandas`, `numpy`, `scikit-learn`
- **Machine Learning**: `LightGBM`, `RandomForest`, `imbalanced-learn` (SMOTE)
- **Hyperparameter Tuning**: `Optuna`
- **Explainable AI (XAI)**: `SHAP`
- **Model Deployment**: `FastAPI`, `Uvicorn`, `joblib`

---

## 📂 Project Structure

1. **Phase1_EDA.py**: Exploratory Data Analysis & extreme imbalance visualization.
2. **Phase2_Baseline.py**: Training baseline Logistic Regression & Random Forest models. Evaluation using **PR-AUC**.
3. **Phase3_Imbalance.py**: Applying **SMOTE** (Synthetic Minority Over-sampling Technique) and Class Weights to penalize False Negatives.
4. **Phase4_GradientBoosting.py**: Building the core ensemble model using **LightGBM** and auto-tuning hyperparameters with **Optuna**.
5. **Phase5_SHAP.py**: Breaking down the black-box model using **SHAP** Summary and Waterfall plots to explain local and global feature importance.
6. **main.py**: The production-ready **FastAPI** application for real-time transaction scoring.

---

## 📊 The Precision-Recall Tradeoff
In fraud detection, standard "Accuracy" is a useless metric (a model that blindly guesses "Normal" every time is 99.8% accurate). Instead, this project focuses heavily on maximizing **Recall** (catching every single fraud) while maintaining a high **PR-AUC** (Precision-Recall Area Under Curve) to prevent too many false alarms.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/aitazazahsan01/Fraud_Detection_System_using_Ensemble-ML_and_Explainable-AI.git
cd Fraud_Detection_System_using_Ensemble-ML_and_Explainable-AI
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn pydantic joblib scikit-learn lightgbm pandas numpy shap imbalanced-learn optuna matplotlib seaborn
```

### 3. Run the ML Pipeline
You can run the Python scripts sequentially to train the model, or use Google Colab to run the training and download the `fraud_model.joblib` file into this directory.

### 4. Start the FastAPI Server
Once the `fraud_model.joblib` is present in the root directory:
```bash
uvicorn main:app --reload
```

### 5. Test the API
Open your browser and navigate to `http://127.0.0.1:8000/docs`. You can use the interactive Swagger UI to send a test JSON payload of 30 transaction features and receive a real-time `APPROVE_TRANSACTION` or `BLOCK_TRANSACTION` response.

---

## 🧠 Explainable AI (SHAP)
This system doesn't just block transactions; it explains them. Using `shap.TreeExplainer`, the system generates:
- **Summary Plots**: Showing the global impact of features like `Amount`, `Time`, and hidden variables (`V1`-`V28`).
- **Waterfall Plots**: Showing exactly how a specific transaction's probability moved from the baseline to a Fraud prediction, ensuring full regulatory transparency.
