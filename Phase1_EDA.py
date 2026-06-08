import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------
# PHASE 1: EDA & Imbalance Analysis
# ---------------------------------------------------------

def run_eda(file_path='creditcard.csv'):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print("\n--- Basic Dataset Info ---")
    print(df.info())
    print("\nTotal missing values in dataset:", df.isnull().sum().max())
    
    print("\n--- Class Distribution ---")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    
    fraud_pct = (class_counts[1] / len(df)) * 100
    print(f"\nFraud accounts for ONLY {fraud_pct:.3f}% of all transactions.")
    print("This extreme imbalance means a 'dumb' model that always predicts 'Normal' ")
    print("will be 99.827% accurate. This is why standard Accuracy is a useless metric here!")

    # 1. Plot Class Imbalance
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Normal, 1: Fraud)')
    plt.yscale('log') # Log scale because of extreme imbalance
    plt.ylabel('Count (Log Scale)')
    plt.show()

    # 2. Transaction Amount Analysis
    print("\nGenerating Amount Distribution plots...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', ax=ax[0])
    ax[0].set_title('Transaction Amount - Normal')
    ax[0].set_yscale('log') # Log scale to see outliers
    
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', ax=ax[1])
    ax[1].set_title('Transaction Amount - Fraud')
    
    plt.tight_layout()
    plt.show()

    # 3. Time Analysis (Time is in seconds from first transaction)
    print("\nGenerating Time Distribution plots...")
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['Class'] == 0]['Time'], color='blue', label='Normal', fill=True, alpha=0.3)
    sns.kdeplot(df[df['Class'] == 1]['Time'], color='red', label='Fraud', fill=True, alpha=0.3)
    plt.title('Density of Transactions over Time (Normal vs Fraud)')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # If running locally, ensure 'creditcard.csv' is in the same directory.
    # In Colab, you can just paste the contents of this function.
    run_eda('creditcard.csv')
