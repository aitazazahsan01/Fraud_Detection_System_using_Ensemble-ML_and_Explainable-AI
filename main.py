from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Real-Time Fraud Detection API", 
    description="Scores transactions for fraud in < 50ms using a tuned LightGBM model."
)

# Load the Trained Model
try:
    # This expects the model you downloaded from Colab to be in the same directory
    model = joblib.load('fraud_model.joblib')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    model = None
    print("⚠️ WARNING: 'fraud_model.joblib' not found. Please download it from Colab and place it in this folder.")

# Define the expected JSON payload
class Transaction(BaseModel):
    # Expecting exactly 30 features (Time, V1-V28, Amount)
    features: list[float]

@app.get("/")
def health_check():
    return {"status": "API is running. Use POST /predict to score a transaction."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(transaction.features)}")
        
    # Convert list to 2D numpy array for LightGBM
    features_array = np.array(transaction.features).reshape(1, -1)
    
    # Get Probability of Fraud (Class 1)
    fraud_prob = float(model.predict_proba(features_array)[0][1])
    
    # We use 0.5 as the threshold, but in banking you might lower this to 0.3 to catch more fraud
    is_fraud = bool(fraud_prob > 0.5)
    
    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "action": "BLOCK_TRANSACTION" if is_fraud else "APPROVE_TRANSACTION"
    }

if __name__ == "__main__":
    print("Starting FastAPI Server...")
    uvicorn.run("main:app", host="127.0.0.0", port=8000, reload=True)
