import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def run_imbalance_handling(file_path='creditcard.csv'):
    # Load and prep data
    df = pd.read_csv(file_path).dropna()
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    # Helper Function
    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{'='*40}")
        print(f"--- {model_name} Evaluation ---")
        print(classification_report(y_test, y_pred))
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC: {pr_auc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    # 1. SMOTE
    print("\n--- Applying SMOTE to Training Data ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original Training shape: {X_train_scaled.shape}, Frauds: {sum(y_train == 1)}")
    print(f"SMOTE Training shape: {X_train_smote.shape}, Frauds: {sum(y_train_smote == 1)}")

    # 2. Train with SMOTE
    print("\nTraining Random Forest on SMOTE data...")
    rf_smote = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_smote, y_train_smote)
    evaluate_model(rf_smote, X_test_scaled, y_test, "Random Forest (SMOTE)")

    # 3. Train with Class Weights
    print("\nTraining Random Forest with Class Weights (No SMOTE)...")
    rf_weighted = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_weighted.fit(X_train_scaled, y_train)
    evaluate_model(rf_weighted, X_test_scaled, y_test, "Random Forest (Class Weights)")

if __name__ == "__main__":
    run_imbalance_handling()
