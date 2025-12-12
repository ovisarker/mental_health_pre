import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Page Config
st.set_page_config(page_title="Student Mental Health AI", page_icon="🧠", layout="wide")

# 1. Load Resources
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('mental_health_hybrid_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        # Load columns logic
        try:
            columns = joblib.load('feature_columns.pkl')
        except:
            columns = None
        return model, encoders, columns
    except Exception as e:
        return None, None, None

model, encoders, feature_columns = load_resources()

if model is None:
    st.error("🚨 Model files not found! Please make sure .pkl files are uploaded to GitHub.")
    st.stop()

# Helper: Extract numbers safely
def extract_number(text):
    try:
        if pd.isna(text): return 0.0
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
        return float(match.group()) if match else 0.0
    except:
        return 0.0

# Header
st.title("🧠 AI-Powered Student Mental Health Assessment")
st.markdown("This system uses a **Hybrid Machine Learning Model** to analyze patterns from your inputs. No manual scoring is involved.")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("📝 Student Profile")
age_input = st.sidebar.selectbox("1. Age Group", ['18-22', '23-26', '27-30', 'Above 30'])
gender = st.sidebar.selectbox("2. Gender", ['Male', 'Female'])
uni = st.sidebar.selectbox("3. University Type", ['Public', 'Private'])
dept = st.sidebar.text_input("4. Department", "CSE")
year = st.sidebar.selectbox("5. Academic Year", ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'])
cgpa_input = st.sidebar.text_input("6. Current CGPA", "3.50")
scholarship = st.sidebar.selectbox("7. Scholarship/Waiver?", ['Yes', 'No'])

# --- MAIN FORM ---
st.subheader("📋 Self-Assessment")
st.info("Select the option that best describes your feeling.")

# Clean Options Map (No 0, 1, 2, 3 visible to user)
options_map = {
    "Not at all / Never": 0,
    "Several days / Sometimes": 1,
    "More than half the days / Often": 2,
    "Nearly every day / Very Often": 3
}

questions = [
    "Q1. Felt upset due to academic affairs?", "Q2. Unable to control important things?", "Q3. Felt nervous and stressed?",
    "Q4. Could not cope with mandatory activities?", "Q5. Confident about handling problems?", "Q6. Things going your way?",
    "Q7. Able to control irritations?", "Q8. Felt academic performance was on top?", "Q9. Angered due to bad performance?",
    "Q10. Difficulties piling up so high?", 
    "Q11. Feeling nervous, anxious or on edge?", "Q12. Unable to stop worrying?", "Q13. Trouble relaxing?",
    "Q14. Being so restless?", "Q15. Easily annoyed or irritable?", "Q16. Afraid something awful might happen?",
    "Q17. Worrying too much?", 
    "Q18. Little interest or pleasure in doing things?", "Q19. Feeling down, depressed, or hopeless?", "Q20. Trouble falling/staying asleep?",
    "Q21. Feeling tired or having little energy?", "Q22. Poor appetite or overeating?", "Q23. Feeling bad about yourself?",
    "Q24. Trouble concentrating?", "Q25. Moving/speaking slowly or too fast?", "Q26. Thoughts of hurting yourself?"
]

answers = []
cols = st.columns(2)

for i, q in enumerate(questions):
    with cols[i % 2]:
        # User sees text, Code gets number
        val = st.selectbox(q, list(options_map.keys()), key=i)
        answers.append(options_map[val])

# --- PREDICTION ---
st.divider()
if st.button("🚀 Analyze My Mental Health", type="primary"):
    
    # Preprocessing
    age_numeric = extract_number(age_input)
    cgpa_numeric = extract_number(cgpa_input)
    
    raw_input = [age_numeric, gender, uni, dept, year, cgpa_numeric, scholarship] + answers
    
    # Column matching
    if feature_columns is None:
         feature_columns = ['1. Age', '2. Gender', '3. University', '4. Department', '5. Academic Year', '6. Current CGPA', '7. Did you receive a waiver or scholarship at your university?'] + \
                          [f'Q{i+1}' for i in range(26)]

    input_df = pd.DataFrame([raw_input], columns=feature_columns)
    
    # Inference
    try:
        with st.spinner("AI Model is analyzing your patterns..."):
            probs = model.predict_proba(input_df)
        
        st.subheader("📊 AI Diagnosis Result")
        
        result_cols = st.columns(3)
        conditions = ['Anxiety', 'Stress', 'Depression']
        risk_scores = []
        
        for i, cond in enumerate(conditions):
            prob_arr = probs[i][0]
            best_idx = np.argmax(prob_arr)
            label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
            confidence = prob_arr[best_idx] * 100
            
            risk_scores.append((cond, confidence))
            
            with result_cols[i]:
                # Dynamic Color based on severity
                color = "red" if "Severe" in label or "High" in label else "green"
                st.markdown(f"**{cond}**")
                st.markdown(f":{color}[{label}]")
                st.progress(int(confidence))

        dominant = max(risk_scores, key=lambda x: x[1])
        st.error(f"🚨 **Primary Issue:** {dominant[0]} ({dominant[1]:.1f}%)")
        
    except Exception as e:
        st.error(f"Something went wrong: {e}")
