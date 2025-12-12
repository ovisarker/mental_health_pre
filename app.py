import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Page Config
st.set_page_config(page_title="Student Mental Health AI", page_icon="🧠", layout="wide")

# 1. Load Resources (Cached for performance)
@st.cache_resource
def load_resources():
    model = joblib.load('mental_health_hybrid_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    # Load feature columns to ensure correct order
    try:
        columns = joblib.load('feature_columns.pkl')
    except:
        columns = None
    return model, encoders, columns

try:
    model, encoders, feature_columns = load_resources()
    st.success("✅ System Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Helper Function to Extract Number from Text
def extract_number(text):
    try:
        # If text is like "18-22", take "18"
        if '-' in str(text):
            return float(str(text).split('-')[0].strip())
        # If text is just a number string "3.50"
        return float(re.findall(r"[-+]?\d*\.\d+|\d+", str(text))[0])
    except:
        return 0.0

# Header
st.title("🧠 AI-Powered Student Mental Health Assessment")
st.markdown("""
This system uses a **Hybrid Machine Learning Model** (Random Forest + Gradient Boosting) to analyze student mental health patterns.
""")

st.divider()

# --- SIDEBAR: DEMOGRAPHICS ---
st.sidebar.header("📝 Student Profile")

# Age needs to be converted to number later
age_input = st.sidebar.selectbox("1. Age Group", ['18-22', '23-26', '27-30', 'Above 30'])
gender = st.sidebar.selectbox("2. Gender", ['Male', 'Female'])
uni = st.sidebar.selectbox("3. University Type", ['Public', 'Private'])
dept = st.sidebar.text_input("4. Department (e.g., CSE)", "CSE")
year = st.sidebar.selectbox("5. Academic Year", ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'])
# CGPA needs to be converted to number later
cgpa_input = st.sidebar.text_input("6. Current CGPA", "3.50")
scholarship = st.sidebar.selectbox("7. Received Scholarship?", ['Yes', 'No'])

# --- MAIN FORM: 26 QUESTIONS ---
st.header("📋 Self-Assessment Questionnaire")
st.info("Please answer the following 26 questions based on your experience.")

# Options Mapping (Text to Number)
options_map = {
    "0 - Not at all / Never": 0,
    "1 - Several days / Sometimes": 1,
    "2 - More than half the days / Often": 2,
    "3 - Nearly every day / Very Often": 3
}

questions = [
    "Q1. Felt upset due to academic affairs?", "Q2. Unable to control important things?", "Q3. Felt nervous and stressed?",
    "Q4. Could not cope with mandatory activities?", "Q5. Confident about handling problems?", "Q6. Things going your way?",
    "Q7. Able to control irritations?", "Q8. Felt academic performance was on top?", "Q9. Angered due to bad performance?",
    "Q10. Difficulties piling up so high?", 
    "Q11. Feeling nervous, anxious or on edge?", "Q12. Unable to stop worrying?", "Q13. Trouble relaxing?",
    "Q14. Being so restless?", "Q15. Easily annoyed or irritable?", "Q16. Afraid something awful might happen?",
    "Q17. Worrying too much?", 
    "Q18. Little interest or pleasure in doing things?", "Q19. Feeling down, depressed, or hopeless?", "Q20. Trouble falling or staying asleep?",
    "Q21. Feeling tired or having little energy?", "Q22. Poor appetite or overeating?", "Q23. Feeling bad about yourself?",
    "Q24. Trouble concentrating?", "Q25. Moving/speaking slowly or too fast?", "Q26. Thoughts of hurting yourself?"
]

answers = []
cols = st.columns(2)

for i, q in enumerate(questions):
    with cols[i % 2]:
        val = st.selectbox(q, list(options_map.keys()), key=i)
        answers.append(options_map[val])

# --- PREDICTION LOGIC ---
st.divider()
if st.button("🚀 Analyze My Mental Health", type="primary"):
    
    # --- FIX: Convert Numeric Demographics Manually ---
    # Model expects numbers for Age and CGPA because of Median Imputer
    age_numeric = extract_number(age_input)
    cgpa_numeric = extract_number(cgpa_input)
    
    # 1. Prepare Raw Input List (7 Demo + 26 Questions)
    # Order must match training columns
    raw_input = [age_numeric, gender, uni, dept, year, cgpa_numeric, scholarship] + answers
    
    # 2. Create DataFrame with correct column names
    # If feature_columns failed to load, we construct manual names matching training
    if feature_columns is None:
        # Fallback names
        feature_columns = ['1. Age', '2. Gender', '3. University', '4. Department', '5. Academic Year', '6. Current CGPA', '7. Did you receive a waiver or scholarship at your university?'] + \
                          [f'Q{i+1}' for i in range(26)] # Placeholder for question names, risky if not exact match

    input_df = pd.DataFrame([raw_input], columns=feature_columns)
    
    # 3. Predict
    try:
        with st.spinner("Analyzing patterns..."):
            probs = model.predict_proba(input_df)
        
        # 4. Display Results
        st.subheader("📊 AI Diagnosis Report")
        
        result_cols = st.columns(3)
        conditions = ['Anxiety', 'Stress', 'Depression']
        risk_scores = []
        
        for i, cond in enumerate(conditions):
            prob_arr = probs[i][0]
            best_idx = np.argmax(prob_arr)
            
            # Decode Label
            label_key = f"{cond} Label"
            pred_label = encoders[label_key].inverse_transform([best_idx])[0]
            confidence = prob_arr[best_idx] * 100
            
            risk_scores.append((cond, confidence))
            
            with result_cols[i]:
                st.metric(label=cond, value=pred_label, delta=f"{confidence:.1f}% Match")
                st.progress(int(confidence))

        # 5. Dominant Issue
        dominant = max(risk_scores, key=lambda x: x[1])
        st.error(f"🚨 **Primary Detected Issue:** {dominant[0]} ({dominant[1]:.1f}%)")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Debug Info: Column names mismatch or data type issue.")
