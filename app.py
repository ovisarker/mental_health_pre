import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load('mental_health_hybrid_model.pkl')
encoders = joblib.load('label_encoders.pkl')

st.title("🧠 AI Mental Health Assessment System")
st.write("This AI system analyzes your academic stress, anxiety, and depression levels based on your responses.")

# --- SIDEBAR: Demographics ---
st.sidebar.header("User Profile")
age = st.sidebar.selectbox("Age Group", ["18-22", "23-26", "27-30", "Above 30"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
uni = st.sidebar.selectbox("University Type", ["Public", "Private"])
dept = st.sidebar.text_input("Department", "CSE")
year = st.sidebar.selectbox("Academic Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
cgpa = st.sidebar.text_input("Current CGPA", "3.50")
scholarship = st.sidebar.selectbox("Scholarship?", ["Yes", "No"])

# --- MAIN FORM: 26 Questions ---
st.header("Self-Assessment Questionnaire")
st.info("Please answer honestly based on your feelings over the last 2 weeks.")

# Define Questions (Shortened for Demo - You will put all 26)
questions = [
    "1. How often have you felt upset due to academic affairs?",
    "2. How often felt unable to control important things?",
    "3. How often felt nervous and stressed?",
    # ... Add all 26 questions here ...
]

answers = []
options = ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Nearly every day"]

for q in questions:
    ans = st.selectbox(q, options)
    # Convert text to number (0,1,2,3)
    answers.append(int(ans.split('-')[0]))

if st.button("Analyze My Mental Health"):
    # Prepare Input DataFrame
    input_data = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'University': [uni], 
        'Department': [dept], 'Academic Year': [year], 'Current CGPA': [cgpa], 
        'Scholarship': [scholarship]
    })
    
    # Add Question Columns
    for i, val in enumerate(answers):
        input_data[f'Q{i+1}'] = val
        
    # Predict
    probs = model.predict_proba(input_data)
    
    # Display Results
    st.subheader("📊 Your AI Analysis Report")
    
    conditions = ['Anxiety', 'Stress', 'Depression']
    risk_data = []

    for i, cond in enumerate(conditions):
        prob = probs[i][0]
        idx = np.argmax(prob)
        label = encoders[f'{cond} Label'].inverse_transform([idx])[0]
        conf = prob[idx] * 100
        
        st.write(f"**{cond}:** {label} ({conf:.1f}%)")
        st.progress(int(conf))
        risk_data.append((cond, conf))
        
    # Dominant
    dominant = max(risk_data, key=lambda x: x[1])
    st.error(f"🚨 **Main Issue Detected:** {dominant[0]} ({dominant[1]:.1f}%)")
