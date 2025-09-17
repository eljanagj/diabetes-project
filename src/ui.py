import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DiabetesPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_cols = ['gender', 'pregnancies', 'age', 'glucose', 'blood_pressure', 
                            'weight', 'height', 'insulin', 'bmi', 'physically_active', 
                            'smoking', 'junk_food', 'family_history']
        self.best_model_info = None
        self.models_loaded = False
    
    def load_models(self):
        """Load all trained models and scalers"""
        if self.models_loaded:
            return True
            
        try:
            with st.spinner("Loading models..."):
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
                
                self.models_loaded = True
                return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict(self, data, model_name=None):
        """Make prediction using specified model or best model"""
        if not self.load_models():
            return {"error": "Failed to load models"}
            
        if model_name is None:
            model_name = self.best_model_info.get('model_name', 'Random Forest')
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        try:
            input_data = pd.DataFrame([data], columns=self.feature_cols)
            
            if model_name in ['Random Forest', 'GBM']:
                X_processed = input_data
            else:
                X_processed = self.scalers[model_name].transform(input_data)
            
            probability = self.models[model_name].predict_proba(X_processed)[0][1]
            prediction = 1 if probability >= 0.5 else 0
            
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "risk_level": self.get_risk_level(probability),
                "model_used": model_name
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

@st.cache_resource
def load_predictor():
    return DiabetesPredictor()

def main():

    predictor = load_predictor()
    
    if not predictor.load_models():
        st.error("Failed to load models. Please check the model files.")
        return
        st.header("Model Information")
        
        if predictor.best_model_info:
            st.success(f"**Best Model:** {predictor.best_model_info.get('model_name', 'N/A')}")
            st.metric("F1-Score", f"{predictor.best_model_info.get('f1_score', 0):.3f}")
            st.metric("AUC-ROC", f"{predictor.best_model_info.get('auc_roc', 0):.3f}")
            st.metric("Accuracy", f"{predictor.best_model_info.get('accuracy', 0):.3f}")
        
        st.header("Model Information")
        st.info("Using Random Forest (Best Model)")
        
        st.header("About")
        st.info("""
        This system uses machine learning to predict diabetes risk based on:
        - Personal information (age, gender, BMI)
        - Medical history (pregnancies, family history)
        - Lifestyle factors (physical activity, smoking, diet)
        - Clinical measurements (glucose, blood pressure, insulin)
        """)
    
    # Center the form using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        st.title("🩺 Diabetes Risk Prediction")
        with st.form("prediction_form"):
            st.subheader("Personal Information")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            
            st.subheader("Clinical Measurements")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=600, value=100)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=250, value=80)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            insulin = st.number_input("Insulin Level (μU/mL)", min_value=0.0, max_value=1000.0, value=30.0, step=0.1)
            
            st.subheader("Lifestyle Factors")
            physically_active = st.selectbox("Physically Active", ["No", "Yes"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            junk_food = st.selectbox("Junk Food Consumption", ["No", "Yes"])
            family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
            
            bmi = weight / ((height / 100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            submitted = st.form_submit_button("🔍 Assess Diabetes Risk", use_container_width=True)
    
    with col2:
        st.header("Prediction Results")
        
        if submitted:
            data = {
                'gender': 1 if gender == "Male" else 0,
                'pregnancies': pregnancies,
                'age': age,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'weight': weight,
                'height': height,
                'insulin': insulin,
                'bmi': bmi,
                'physically_active': 1 if physically_active == "Yes" else 0,
                'smoking': 1 if smoking == "Yes" else 0,
                'junk_food': 1 if junk_food == "Yes" else 0,
                'family_history': 1 if family_history == "Yes" else 0
            }
            
            with st.spinner("Analyzing data..."):
                result = predictor.predict(data, "Random Forest")
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                probability = result['probability']
                prediction = result['prediction']
                risk_level = result['risk_level']
                
                if risk_level == "Low Risk":
                    color = "green"
                    icon = "✅"
                elif risk_level == "Medium Risk":
                    color = "orange"
                    icon = "⚠️"
                else:
                    color = "red"
                    icon = "🚨"
                
                st.markdown(f"### {icon} {risk_level}")
                
                st.metric("Risk Probability", f"{probability * 100:.1f}%")
                
                st.progress(probability)
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Prediction", "Diabetes" if prediction == 1 else "No Diabetes")
                with col2b:
                    st.metric("Model Used", "Random Forest")
                
                st.subheader("Risk Interpretation")
                if risk_level == "Low Risk":
                    st.success("""
                    **Low Risk** - The model suggests a low probability of diabetes.
                    Continue maintaining a healthy lifestyle with regular exercise
                    and a balanced diet.
                    """)
                elif risk_level == "Medium Risk":
                    st.warning("""
                    **Medium Risk** - The model indicates a moderate probability of diabetes.
                    Consider consulting with a healthcare provider and making
                    lifestyle improvements.
                    """)
                else:
                    st.error("""
                    **High Risk** - The model suggests a high probability of diabetes.
                    Please consult with a healthcare provider immediately for
                    proper evaluation and management.
                    """)
        
        else:
            st.info("Please fill in the patient information and click 'Assess Diabetes Risk' to get started.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
        <p>Always consult with a healthcare provider for proper medical evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
