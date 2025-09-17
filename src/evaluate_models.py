import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')
# import tensorflow as tf  # Not needed without ANN

def load_and_evaluate_models():
    """Load all trained models and evaluate them on test set"""
    
    print("Loading data and preparing test set...")
    
    df = pd.read_csv("data/cleaned_responses.csv")
    
    feature_cols = ['gender', 'pregnancies', 'age', 'glucose', 'blood_pressure', 
                    'weight', 'height', 'insulin', 'bmi', 'physically_active', 
                    'smoking', 'junk_food', 'family_history']
    
    X = df[feature_cols].copy()
    y = df['diabetes_diagnosed'].astype(int).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Test set target distribution:")
    print(y_test.value_counts())
    
    scalers = {
        'Logistic Regression': joblib.load("models/scaler.pkl"),
        'Random Forest': joblib.load("models/scaler_rf.pkl"),
        'SVM': joblib.load("models/scaler_svm.pkl"),
        'GBM': joblib.load("models/scaler_gbm.pkl")
    }
    
    models = {
        'Logistic Regression': joblib.load("models/logistic_regression_model.pkl"),
        'Random Forest': joblib.load("models/random_forest_model.pkl"),
        'SVM': joblib.load("models/svm_model.pkl"),
        'GBM': joblib.load("models/gbm_model.pkl")
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        if name == 'Random Forest' or name == 'GBM':
            X_test_processed = X_test
        else:
            X_test_processed = scalers[name].transform(X_test)
        
        if name == 'ANN':
            y_pred_proba = model.predict(X_test_processed, verbose=0).ravel()
            
        else:
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    best_model_name = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['F1-Score']
    best_auc = results_df.iloc[0]['AUC-ROC']
    
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"AUC-ROC: {best_auc:.4f}")
    print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
    print(f"Precision: {results_df.iloc[0]['Precision']:.4f}")
    print(f"Recall: {results_df.iloc[0]['Recall']:.4f}")
    
    best_model_info = {
        'model_name': best_model_name,
        'f1_score': float(best_f1),
        'auc_roc': float(best_auc),
        'accuracy': float(results_df.iloc[0]['Accuracy']),
        'precision': float(results_df.iloc[0]['Precision']),
        'recall': float(results_df.iloc[0]['Recall'])
    }
    
    with open('models/best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"\nBest model information saved to: models/best_model_info.json")
    
    return results_df, best_model_info

if __name__ == "__main__":
    results_df, best_model_info = load_and_evaluate_models()