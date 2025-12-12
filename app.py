import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Models with Caching
@st.cache_resource
def load_models():
    # Load the files you uploaded to GitHub
    model = joblib.load('mental_health_hybrid_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    # Load feature column names to ensure correct input order
    try:
        features = joblib.load('feature_columns.pkl')
    except:
        # Fallback if file missing: Manual list based on your training
        features = ['Age', 'Gender', 'University', 'Department', 'Academic Year', 'Current CGPA', 'Scholarship'] + [f'Q{i}' for i in range(1, 27)]
    return model, encoders, features

# Page Config
st.set_page_config(page_title="Student Mental Health AI", layout="wide")

try:
    model, encoders, feature_columns = load_models()
except Exception as e:
    st.error(f"Error loading models. Please ensure .pkl files are in GitHub. Details: {e}")
    st.stop()

st.title("🧠 AI-Powered Student Mental Health Assessment")
st.markdown("This system uses a **Hybrid ML Model** (Random Forest + Gradient Boosting) trained on raw student data.")

# --- SIDEBAR: 7 DEMOGRAPHICS ---
st.sidebar.header("Student Profile")
age = st.sidebar.selectbox("1. Age Group", ['18-22', '23-26', '27-30', 'Above 30'])
gender = st.sidebar.selectbox("2. Gender", ['Male', 'Female'])
uni = st.sidebar.selectbox("3. University Type", ['Public', 'Private'])
dept = st.sidebar.text_input("4. Department", "CSE")
year = st.sidebar.selectbox("5. Academic Year", ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'])
cgpa = st.sidebar.text_input("6. Current CGPA", "3.50")
scholarship = st.sidebar.selectbox("7. Scholarship/Waiver?", ['Yes', 'No'])

# --- MAIN FORM: 26 QUESTIONS ---
st.info("Please answer the following 26 questions (Scale: 0 to 3/4)")
questions = [
    "Q1. Upset due to academic affairs?", "Q2. Unable to control important things?", "Q3. Nervous and stressed?", 
    "Q4. Could not cope with mandatory activities?", "Q5. Confident about handling problems?", "Q6. Things going your way?", 
    "Q7. Able to control irritations?", "Q8. Felt academic performance was on top?", "Q9. Angered due to bad performance?", 
    "Q10. Difficulties piling up?", "Q11. Nervous, anxious or on edge?", "Q12. Unable to stop worrying?", 
    "Q13. Trouble relaxing?", "Q14. Being so restless?", "Q15. Easily annoyed or irritable?", 
    "Q16. Afraid something awful might happen?", "Q17. Worrying too much?", "Q18. Little interest in doing things?", 
    "Q19. Feeling down, depressed, or hopeless?", "Q20. Trouble falling/staying asleep?", "Q21. Feeling tired/little energy?", 
    "Q22. Poor appetite or overeating?", "Q23. Feeling bad about yourself?", "Q24. Trouble concentrating?", 
    "Q25. Moving slowly or too fast?", "Q26. Thoughts of hurting yourself?"
]

# Create Input Grid
answers = []
cols = st.columns(3)
for i, q in enumerate(questions):
    with cols[i % 3]:
        val = st.selectbox(q, ["0 - Never/Not at all", "1 - Sometimes", "2 - Often", "3 - Very Often/Nearly every day"], key=i)
        answers.append(int(val.split(' - ')[0]))

# --- PREDICTION ---
if st.button("🚀 Analyze Mental Health", type="primary"):
    # Prepare Input DataFrame
    raw_data = [age, gender, uni, dept, year, cgpa, scholarship] + answers
    input_df = pd.DataFrame([raw_data], columns=feature_columns)
    
    st.write("Processing...")
    
    # Get Probabilities
    probs = model.predict_proba(input_df)
    
    st.subheader("📊 Analysis Results")
    c1, c2, c3 = st.columns(3)
    conditions = ['Anxiety', 'Stress', 'Depression']
    risk_data = []

    for i, cond in enumerate(conditions):
        prob_arr = probs[i][0]
        best_idx = np.argmax(prob_arr)
        
        # Decode Label
        label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
        conf = prob_arr[best_idx] * 100
        risk_data.append((cond, conf))
        
        with [c1, c2, c3][i]:
            st.metric(cond, f"{label}", f"{conf:.1f}% Match")
            st.progress(int(conf))

    dominant = max(risk_data, key=lambda x: x[1])
    st.error(f"🚨 **Primary Issue:** {dominant[0]} ({dominant[1]:.1f}%)")
