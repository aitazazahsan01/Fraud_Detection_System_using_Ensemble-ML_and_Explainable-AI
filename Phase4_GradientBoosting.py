import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def run_gradient_boosting(file_path='creditcard.csv'):
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
        return pr_auc

    # Calculate scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Optuna Objective Function
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        return auc(recall, precision) # We maximize PR-AUC

    # Run Optuna Study
    print("\n--- Running Optuna Hyperparameter Tuning ---")
    # Limiting to 10 trials to save time
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) 
    
    print("\nBest Parameters found by Optuna:")
    print(study.best_params)

    # Train Final Model
    print("\n--- Training Final LightGBM Model ---")
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    final_lgb = lgb.LGBMClassifier(**best_params)
    final_lgb.fit(X_train_scaled, y_train)
    
    evaluate_model(final_lgb, X_test_scaled, y_test, "LightGBM (Tuned)")

if __name__ == "__main__":
    run_gradient_boosting()
