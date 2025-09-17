from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

app = FastAPI(title="Diabetes Risk Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    gender: int
    pregnancies: int
    age: int
    glucose: float
    blood_pressure: float
    weight: float
    height: float
    insulin: float
    bmi: float
    physically_active: int
    smoking: int
    junk_food: int
    family_history: int
    model_name: str = None

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    model_used: str

class DiabetesPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cols = ['gender', 'pregnancies', 'age', 'glucose', 'blood_pressure', 
                            'weight', 'height', 'insulin', 'bmi', 'physically_active', 
                            'smoking', 'junk_food', 'family_history']
        self.load_models()
    
    def load_models(self):
        """Load all trained models and scalers"""
        try:
            self.models['Logistic Regression'] = joblib.load('models/logistic_regression_model.pkl')
            self.models['Random Forest'] = joblib.load('models/random_forest_model.pkl')
            self.models['SVM'] = joblib.load('models/svm_model.pkl')
            self.models['GBM'] = joblib.load('models/gbm_model.pkl')
            
            
            self.scalers['Logistic Regression'] = joblib.load('models/scaler.pkl')
            self.scalers['Random Forest'] = joblib.load('models/scaler_rf.pkl')
            self.scalers['SVM'] = joblib.load('models/scaler_svm.pkl')
            self.scalers['GBM'] = joblib.load('models/scaler_gbm.pkl')
            
            with open('models/best_model_info.json', 'r') as f:
                self.best_model_info = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict(self, data, model_name=None):
        """Make prediction using specified model or best model"""
        if model_name is None:
            model_name = self.best_model_info.get('model_name', 'Random Forest')
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        input_data = pd.DataFrame([data], columns=self.feature_cols)
        
        if model_name in ['Random Forest', 'GBM']:
            X_processed = input_data
        else:
            X_processed = self.scalers[model_name].transform(input_data)
        
        if model_name == 'ANN':
            probability = self.models[model_name].predict(X_processed, verbose=0)[0][0]
        else:
            probability = self.models[model_name].predict_proba(X_processed)[0][1]
        
        prediction = 1 if probability >= 0.5 else 0
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": self.get_risk_level(probability),
            "model_used": model_name
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

predictor = DiabetesPredictor()

@app.get("/")
async def root():
    return {"message": "Diabetes Risk Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(predictor.models),
        "best_model": predictor.best_model_info.get('model_name', 'N/A')
    }

@app.get("/models")
async def get_models():
    """Get available models and their performance"""
    models_info = []
    for name in predictor.models.keys():
        if name == predictor.best_model_info.get('model_name'):
            models_info.append({
                "name": name,
                "is_best": True,
                "f1_score": predictor.best_model_info.get('f1_score', 0),
                "auc_roc": predictor.best_model_info.get('auc_roc', 0)
            })
        else:
            models_info.append({
                "name": name,
                "is_best": False
            })
    
    return {
        "models": models_info,
        "best_model": predictor.best_model_info.get('model_name', 'N/A')
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_diabetes(request: PredictionRequest):
    """Make diabetes risk prediction"""
    try:
        # Convert request to dict
        data = request.dict()
        model_name = data.pop('model_name', None)
        
        # Make prediction
        result = predictor.predict(data, model_name)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)