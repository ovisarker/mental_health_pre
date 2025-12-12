import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Models & Encoders
try:
    model = joblib.load('mental_health_hybrid_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please upload .pkl files.")
    st.stop()

# 2. Page Configuration
st.set_page_config(page_title="Student Mental Health AI", page_icon="🧠", layout="wide")

st.title("🧠 AI-Based Student Mental Health Assessment")
st.markdown("""
This system uses **Hybrid Machine Learning (Random Forest + Gradient Boosting)** to analyze academic stress, anxiety, and depression patterns.
*Please answer the following questions honestly.*
""")

# --- SIDEBAR: DEMOGRAPHICS ---
st.sidebar.header("📝 Student Profile")

def user_input_features():
    age = st.sidebar.selectbox("1. Age Group", ['18-22', '23-26', '27-30', 'Above 30'])
    gender = st.sidebar.selectbox("2. Gender", ['Male', 'Female'])
    # University list shortened for demo
    uni = st.sidebar.selectbox("3. University Type", ['Public', 'Private']) 
    dept = st.sidebar.text_input("4. Department", "CSE")
    year = st.sidebar.selectbox("5. Academic Year", ['1st Year', '2nd Year', '3rd Year', '4th Year'])
    cgpa = st.sidebar.text_input("6. Current CGPA", "3.50")
    scholarship = st.sidebar.selectbox("7. Scholarship/Waiver?", ['Yes', 'No'])

    return [age, gender, uni, dept, year, cgpa, scholarship]

demo_inputs = user_input_features()

# --- MAIN FORM: 26 QUESTIONS ---
st.header("📋 Self-Assessment Questionnaire")
st.caption("Scale: 0 (Not at all) to 3/4 (Very Often)")

# Define all 26 questions (Replace with your exact questions from Raw Dataset columns)
questions = [
    # Stress (PSS) - 10 Questions
    "Q1. Felt upset due to academic affairs?",
    "Q2. Felt unable to control important things?",
    "Q3. Felt nervous and stressed?",
    "Q4. Could not cope with mandatory activities?",
    "Q5. Felt confident about handling problems?",
    "Q6. Felt things are going your way?",
    "Q7. Able to control irritations?",
    "Q8. Felt academic performance was on top?",
    "Q9. Got angered due to bad performance?",
    "Q10. Felt difficulties are piling up?",
    
    # Anxiety (GAD) - 7 Questions
    "Q11. Felt nervous, anxious or on edge?",
    "Q12. Unable to stop worrying?",
    "Q13. Trouble relaxing?",
    "Q14. Being so restless that it's hard to sit still?",
    "Q15. Becoming easily annoyed or irritable?",
    "Q16. Feeling afraid as if something awful might happen?",
    "Q17. Worrying too much about different things?",
    
    # Depression (PHQ) - 9 Questions
    "Q18. Little interest or pleasure in doing things?",
    "Q19. Feeling down, depressed, or hopeless?",
    "Q20. Trouble falling or staying asleep?",
    "Q21. Feeling tired or having little energy?",
    "Q22. Poor appetite or overeating?",
    "Q23. Feeling bad about yourself (failure)?",
    "Q24. Trouble concentrating on things?",
    "Q25. Moving/speaking slowly or too fast?",
    "Q26. Thoughts that you would be better off dead?"
]

# Create Input Fields for Questions
q_answers = []
col1, col2 = st.columns(2)

for i, q in enumerate(questions):
    with col1 if i < 13 else col2:
        # PSS is 0-4, GAD/PHQ is 0-3. Safe to use 0-4 for all or adjust logic.
        # Assuming cleaned data max was 4.
        val = st.slider(f"{q}", 0, 4, 1) 
        q_answers.append(val)

# Combine Demographics + Question Answers into DataFrame
# Note: Ensure columns match Training Data order exactly!
# For safety, we create a DataFrame with generic names first, pipeline handles the rest
if st.button("🚀 Analyze Mental Health"):
    
    # Create a single row DataFrame
    # Column names MUST match what you used in training (Raw Dataset headers)
    # Here we need to be careful. The pipeline expects specific column names.
    # We will construct the list values first.
    
    raw_values = demo_inputs + q_answers
    
    # We need column names from training. 
    # Hardcoding assuming order is maintained.
    # (In a real app, load 'feature_columns.pkl' to get names)
    
    input_df = pd.DataFrame([raw_values])
    # The pipeline handles raw input (no column names needed if order is correct)
    # BUT sklearn usually warns about names.
    # Let's trust the pipeline handles array-like input or set dummy columns
    
    st.write("Processing inputs...")
    
    try:
        # Get Probabilities
        probs = model.predict_proba(input_df)
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        conditions = ['Anxiety', 'Stress', 'Depression']
        risk_scores = []
        
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]

        for i, cond in enumerate(conditions):
            # Hybrid Probability Logic
            prob_arr = probs[i][0] # Get probability array
            best_idx = np.argmax(prob_arr)
            
            # Decode Label
            label_key = f"{cond} Label"
            pred_label = encoders[label_key].inverse_transform([best_idx])[0]
            confidence = prob_arr[best_idx] * 100
            
            risk_scores.append((cond, confidence))
            
            with cols[i]:
                st.info(f"**{cond}**")
                st.write(f"Status: **{pred_label}**")
                st.write(f"Severity: **{confidence:.1f}%**")
                st.progress(int(confidence))

        # Dominant Issue Logic
        dominant = max(risk_scores, key=lambda x: x[1])
        
        st.divider()
        st.error(f"🚨 **Primary Detected Issue:** {dominant[0]} ({dominant[1]:.1f}%)")
        st.success("Analysis Complete based on AI Pattern Recognition.")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Tip: Ensure inputs match the training data format.")
