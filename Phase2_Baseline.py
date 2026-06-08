import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def run_baseline_models(file_path='creditcard.csv'):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 1. Train-Test Split
    # stratify=y is CRITICAL so both train and test sets have 0.17% fraud
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Scale Features
    # V1-V28 are already scaled (PCA), but Time and Amount are not.
    print("Scaling 'Time' and 'Amount' features...")
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    # Helper Function for Evaluation
    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{'='*40}")
        print(f"--- {model_name} Evaluation ---")
        print(f"{'='*40}")
        print(classification_report(y_test, y_pred))
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    # 3. Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression Baseline")

    # 4. Train Random Forest (Restricted size to avoid Colab RAM crash)
    print("\nTraining Random Forest (n_estimators=50, max_depth=10)...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest Baseline")

if __name__ == "__main__":
    run_baseline_models('creditcard.csv')
