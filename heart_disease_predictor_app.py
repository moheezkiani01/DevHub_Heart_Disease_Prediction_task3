import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set up the page configuration
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("""
This application evaluates your risk of heart disease based on various health indicators. 
Answer the questions below to get an assessment of your heart disease risk.
""")

# Create sample heart disease data for training
@st.cache_data
def create_sample_heart_data():
    np.random.seed(42)
    
    # Generate synthetic heart disease dataset
    n_samples = 1000
    
    # Features based on real heart disease datasets
    age = np.random.randint(29, 78, n_samples)
    sex = np.random.randint(0, 2, n_samples)  # 0 = female, 1 = male
    chest_pain_type = np.random.randint(0, 4, n_samples)  # 0-3
    resting_bp = np.random.randint(90, 200, n_samples)  # Resting blood pressure
    cholesterol = np.random.randint(120, 560, n_samples)  # Cholesterol level
    fasting_bs = np.random.randint(0, 2, n_samples)  # Fasting blood sugar > 120 mg/dl
    resting_ecg = np.random.randint(0, 3, n_samples)  # Resting ECG results
    max_hr = np.random.randint(60, 202, n_samples)  # Maximum heart rate
    exercise_angina = np.random.randint(0, 2, n_samples)  # Exercise induced angina
    oldpeak = np.random.uniform(0, 6.2, n_samples)  # ST depression
    st_slope = np.random.randint(0, 3, n_samples)  # ST slope
    
    # Create target variable with realistic relationships
    # Higher risk factors increase probability of heart disease
    risk_score = (
        (age - 29) / 50 * 0.3 +  # Age factor
        (chest_pain_type / 3) * 0.2 +  # Chest pain factor
        (resting_bp - 90) / 110 * 0.15 +  # Blood pressure factor
        (cholesterol - 120) / 440 * 0.15 +  # Cholesterol factor
        fasting_bs * 0.1 +  # Fasting blood sugar
        (max_hr - 60) / 142 * -0.2 +  # Max heart rate (inverse relationship)
        exercise_angina * 0.2 +  # Exercise angina
        oldpeak / 6.2 * 0.3  # ST depression
    )
    
    # Convert to binary outcome with some noise
    heart_disease = (risk_score + np.random.normal(0, 0.1, n_samples)) > 0.5
    heart_disease = heart_disease.astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope,
        'HeartDisease': heart_disease
    })
    
    return df

# Train models
@st.cache_resource
def train_models(df):
    # Prepare the data
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression model
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train_scaled, y_train)
    
    # Train Decision Tree model
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred_log = log_reg_model.predict(X_test_scaled)
    y_pred_proba_log = log_reg_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_dt = dt_model.predict(X_test_scaled)
    y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]
    
    log_accuracy = accuracy_score(y_test, y_pred_log)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    log_auc = roc_auc_score(y_test, y_pred_proba_log)
    dt_auc = roc_auc_score(y_test, y_pred_proba_dt)
    
    return log_reg_model, dt_model, scaler, log_accuracy, dt_accuracy, log_auc, dt_auc, X_test_scaled, y_test

# Load data and models
df = create_sample_heart_data()
log_reg_model, dt_model, scaler, log_accuracy, dt_accuracy, log_auc, dt_auc, X_test_scaled, y_test = train_models(df)

# Sidebar for inputs
st.sidebar.header("Health Indicators")

# Input widgets
age = st.sidebar.slider("Age", 20, 90, 55)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=0)
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", 
                                      ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], 
                                      index=0)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0)
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 200, 150)
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"], index=0)
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)

# Convert categorical inputs to numerical
sex_num = 1 if sex == "Male" else 0
chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
chest_pain_num = chest_pain_map[chest_pain_type]
fasting_bs_num = 1 if fasting_bs == "Yes" else 0
exercise_angina_num = 1 if exercise_angina == "Yes" else 0

# Create input dataframe (using average values for non-user inputs)
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_num],
    'ChestPainType': [chest_pain_num],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs_num],
    'RestingECG': [1],  # Using average value
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina_num],
    'Oldpeak': [oldpeak],
    'ST_Slope': [1]  # Using average value
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make predictions
log_pred = log_reg_model.predict(input_scaled)[0]
log_proba = log_reg_model.predict_proba(input_scaled)[0][1]  # Probability of heart disease
dt_pred = dt_model.predict(input_scaled)[0]
dt_proba = dt_model.predict_proba(input_scaled)[0][1]  # Probability of heart disease

# Calculate average probability
avg_proba = (log_proba + dt_proba) / 2

# Display risk assessment
st.subheader("Risk Assessment")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Logistic Regression Risk", 
        value=f"{log_proba*100:.1f}%",
        delta=f"{(log_proba - avg_proba)/avg_proba*100:.1f}% vs Average"
    )

with col2:
    st.metric(
        label="Decision Tree Risk", 
        value=f"{dt_proba*100:.1f}%",
        delta=f"{(dt_proba - avg_proba)/avg_proba*100:.1f}% vs Average"
    )

# Determine risk level
risk_level = ""
if avg_proba < 0.3:
    risk_level = "Low Risk"
    risk_color = "green"
elif avg_proba < 0.6:
    risk_level = "Moderate Risk"
    risk_color = "orange"
else:
    risk_level = "High Risk"
    risk_color = "red"

# Display risk level
st.markdown(f"## :{risk_color}[**Overall Risk Level: {risk_level}**]")

# Display input details
st.subheader("Your Health Information")
input_df = pd.DataFrame({
    'Indicator': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                  'Fasting BS >120 mg/dl', 'Max HR', 'Exercise Angina', 'ST Depression'],
    'Value': [age, str(sex), str(chest_pain_type), resting_bp, cholesterol,
              str(fasting_bs), max_hr, str(exercise_angina), oldpeak]
})
st.dataframe(input_df.reset_index(drop=True))

# Model information
st.subheader("Model Information")
st.write(f"""
- **Logistic Regression** Accuracy: {log_accuracy:.4f} (AUC: {log_auc:.4f})
- **Decision Tree** Accuracy: {dt_accuracy:.4f} (AUC: {dt_auc:.4f})
""")

# Risk explanation
st.subheader("Understanding Your Risk")
if risk_level == "Low Risk":
    st.success("""
    Your risk assessment indicates a lower probability of heart disease based on the provided information.
    However, it's important to maintain a healthy lifestyle and continue regular checkups with your healthcare provider.
    """)
elif risk_level == "Moderate Risk":
    st.warning("""
    Your risk assessment indicates a moderate probability of heart disease.
    Consider discussing these results with your healthcare provider and evaluating lifestyle changes that may reduce your risk.
    """)
else:
    st.error("""
    Your risk assessment indicates a higher probability of heart disease.
    It's important to discuss these results with your healthcare provider as soon as possible to evaluate your health status.
    """)
    
    st.markdown("""
    **Recommendations if you're at high risk:**
    - Schedule a comprehensive health check with a cardiologist
    - Consider lifestyle modifications (diet, exercise, stress management)
    - Monitor your blood pressure and cholesterol regularly
    - Discuss any symptoms with your doctor
    """)

# Feature importance explanation
st.subheader("Key Risk Factors")
st.write("""
Based on the models, the following factors significantly influence heart disease risk:
- **Age**: Risk increases with age
- **Chest Pain Type**: Certain types indicate higher risk
- **Resting Blood Pressure**: High BP increases risk
- **Cholesterol Levels**: High cholesterol increases risk
- **Maximum Heart Rate**: Lower max HR during exercise may indicate issues
- **Exercise-Induced Angina**: Chest pain during exercise indicates risk
- **ST Depression**: Measured during exercise stress test
""")

st.sidebar.markdown("---")
st.sidebar.info("❤️ Heart Disease Risk Assessment\n\nBased on machine learning models trained on health indicators.")

# Health tips
st.subheader("Heart-Healthy Tips")
st.write("""
- **Exercise regularly**: Aim for at least 150 minutes of moderate-intensity exercise per week
- **Eat a balanced diet**: Focus on fruits, vegetables, whole grains, and lean proteins
- **Manage stress**: Practice relaxation techniques like meditation or deep breathing
- **Avoid smoking**: If you smoke, seek help to quit
- **Limit alcohol**: Drink in moderation if at all
- **Regular checkups**: Maintain regular visits with your healthcare provider
""")

# Important disclaimer
st.warning("""
**Important Disclaimer**: This assessment is for educational purposes only and not a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with qualified healthcare providers for any health concerns.
""")

# Footer
st.markdown("---")
st.markdown("Developed as part of AI/ML Engineering Internship Tasks")
st.markdown("This tool uses machine learning to estimate risk based on health indicators but cannot replace professional medical evaluation.")