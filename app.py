import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(page_title="Student Mental Health AI", page_icon="🧠", layout="wide")

# --- RESET LOGIC ---
if 'reset' not in st.session_state:
    st.session_state.reset = False

def reset_app():
    st.session_state.reset = True
    st.rerun()

# 1. Load Resources (Cached)
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('mental_health_hybrid_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, encoders, feature_columns
    except Exception as e:
        return None, None, None

model, encoders, feature_columns = load_resources()

if model is None:
    st.error("🚨 Model files missing! Please upload .pkl files to GitHub.")
    st.stop()

# Helper: Extract numbers safely from text (e.g., "18-22" -> 18.0)
def extract_number(text):
    try:
        if pd.isna(text): return 0.0
        text_str = str(text)
        if '-' in text_str:
            return float(text_str.split('-')[0].strip())
        match = re.search(r"[-+]?\d*\.\d+|\d+", text_str)
        return float(match.group()) if match else 0.0
    except:
        return 0.0

# Helper: Wellness Tips
def get_recommendations(condition):
    tips = {
        "Anxiety": [
            "🌬️ **Deep Breathing:** Try the 4-7-8 breathing technique (Inhale 4s, Hold 7s, Exhale 8s).",
            "🧘 **Grounding:** Name 5 things you see, 4 you feel, 3 you hear.",
            "☕ **Limit Caffeine:** Reduce coffee/tea intake as it fuels anxiety."
        ],
        "Stress": [
            "📝 **Prioritize:** Make a to-do list and break tasks into small steps.",
            "🚶 **Move:** A 10-minute walk can lower cortisol levels.",
            "💤 **Sleep:** Ensure you get 7-8 hours of quality sleep."
        ],
        "Depression": [
            "🤝 **Connect:** Talk to a friend or family member today.",
            "🌞 **Sunlight:** Spend 15 minutes outside in morning light.",
            "📅 **Routine:** Stick to a small, manageable daily routine."
        ],
        "Healthy": [
            "🎉 **Keep going!** Your mental resilience is strong.",
            "💧 **Stay Hydrated:** Drink enough water.",
            "📖 **Journal:** Write down 3 good things that happened today."
        ]
    }
    return tips.get(condition, [])

# --- HEADER & RESET BUTTON ---
col1, col2 = st.columns([8, 2])
with col1:
    st.title("🧠 AI-Powered Student Mental Health Assessment")
with col2:
    if st.button("🔄 Reset All", type="primary"):
        reset_app()

st.markdown("""
This system uses a **Hybrid Machine Learning Model** (Random Forest + Gradient Boosting) trained on raw student data. 
It analyzes behavioral patterns to predict mental health conditions. **No manual scoring or rule-based calculation is used.**
""")

# ==========================================
# 🛠️ SYSTEM ARCHITECTURE (FOR TEACHERS/BOARD)
# ==========================================
with st.expander("ℹ️ Technical Architecture: How this App Works (Backend Logic)"):
    st.markdown("""
    ### **System Workflow & ML Pipeline**
    This section explains how the **Frontend (UI)** connects to the **Backend (AI Engine)** for real-time prediction.

    #### **Step 1: Frontend Layer (User Input)**
    * **Data Collection:** The user provides raw inputs via the Streamlit Interface (7 Demographics + 26 Psychometric Questions).
    * **No Calculation:** The frontend **does NOT** calculate any scores. It collects raw text data (e.g., *"Very Often"*).

    #### **Step 2: Backend Preprocessing (Transformation Layer)**
    * **Text-to-Numeric Mapping:** A backend script converts text inputs into numerical values using Regex (e.g., *"18-22"* $\\rightarrow$ `18.0`).
    * **Feature Alignment:** The system maps the inputs to the exact **33 Feature Columns** used during training.
    
    #### **Step 3: AI Inference Engine (The Core)**
    * **Model:** We use a pre-trained **Hybrid Ensemble Model** combining **Random Forest** and **Gradient Boosting**.
    * **Process:** The model analyzes the input pattern against 2000+ student records to predict probabilities.

    #### **Step 4: Decision Layer (Emotion Detection)**
    * **Probability Decoding:** The system extracts confidence scores and maps them to human-readable labels (e.g., "Severe").
    * **Dominant Issue Logic:** It compares confidence scores to identify the **Primary Mental Health Issue**.
    """)

st.divider()

# --- SIDEBAR: DEMOGRAPHICS ---
st.sidebar.header("📝 Student Profile")

def get_index(options, default_idx=0):
    return 0 if st.session_state.reset else default_idx

age_input = st.sidebar.selectbox("1. Age Group", ['18-22', '23-26', '27-30', 'Above 30'], index=get_index(4))
gender = st.sidebar.selectbox("2. Gender", ['Male', 'Female'], index=get_index(2))
uni = st.sidebar.selectbox("3. University Type", ['Public', 'Private'], index=get_index(2))
dept = st.sidebar.text_input("4. Department", value="" if st.session_state.reset else "CSE")
year = st.sidebar.selectbox("5. Academic Year", ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'], index=get_index(5))
cgpa_input = st.sidebar.text_input("6. Current CGPA", value="" if st.session_state.reset else "3.50")
scholarship = st.sidebar.selectbox("7. Scholarship/Waiver?", ['Yes', 'No'], index=get_index(2))

# --- MAIN FORM ---
st.subheader("📋 Self-Assessment Questionnaire")
st.info("Select options that describe your feelings over the last 2 weeks.")

options_map = {
    "Not at all / Never": 0,
    "Several days / Sometimes": 1,
    "More than half the days / Often": 2,
    "Nearly every day / Very Often": 3
}

q_labels = [
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

answers = []
cols = st.columns(2)
key_modifier = str(st.session_state.reset)

for i, q in enumerate(q_labels):
    with cols[i % 2]:
        idx = 0 if st.session_state.reset else 0
        val = st.selectbox(q, list(options_map.keys()), index=idx, key=f"q_{i}_{key_modifier}")
        answers.append(options_map[val])

if st.session_state.reset:
    st.session_state.reset = False

# --- PREDICTION ---
st.divider()
analyze_btn = st.button("🚀 Analyze My Mental Health", type="primary")

if analyze_btn:
    # 1. Preprocessing
    age_numeric = extract_number(age_input)
    cgpa_numeric = extract_number(cgpa_input)
    
    # 2. Map Inputs
    input_dict = {}
    if len(feature_columns) == 33:
        input_dict[feature_columns[0]] = age_numeric
        input_dict[feature_columns[1]] = gender
        input_dict[feature_columns[2]] = uni
        input_dict[feature_columns[3]] = dept
        input_dict[feature_columns[4]] = year
        input_dict[feature_columns[5]] = cgpa_numeric
        input_dict[feature_columns[6]] = scholarship
        for i in range(26):
            input_dict[feature_columns[7+i]] = answers[i]
            
        input_df = pd.DataFrame([input_dict])
        
        try:
            with st.spinner("AI Model is analyzing patterns..."):
                probs = model.predict_proba(input_df)
            
            st.subheader("📊 AI Diagnosis Result")
            
            result_cols = st.columns(3)
            conditions = ['Anxiety', 'Stress', 'Depression']
            risk_scores = []
            healthy_count = 0
            
            for i, cond in enumerate(conditions):
                prob_arr = probs[i][0]
                best_idx = np.argmax(prob_arr)
                label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                confidence = prob_arr[best_idx] * 100
                
                # Logic: Healthy if label indicates minimal/low/none
                is_healthy = any(safe in label for safe in ["Minimal", "Low", "None", "No Depression"])
                
                with result_cols[i]:
                    st.markdown(f"**{cond}**")
                    if is_healthy:
                        st.success(f"✅ {label}")
                        st.caption(f"Confidence: {confidence:.1f}%")
                        risk_scores.append((cond, 0))
                        healthy_count += 1
                    else:
                        st.error(f"⚠️ {label}")
                        st.progress(int(confidence))
                        st.caption(f"Severity: {confidence:.1f}%")
                        risk_scores.append((cond, confidence))

            st.divider()
            
            # --- VISUALIZATION: RADAR CHART ---
            st.subheader("📈 Emotional Balance Map")
            # Prepare data: Use risk scores (if 0, we show small value for visuals)
            viz_scores = [score if score > 0 else 10 for _, score in risk_scores]
            
            df_chart = pd.DataFrame({
                'Condition': conditions,
                'Risk Level': viz_scores
            })
            
            fig = px.line_polar(df_chart, r='Risk Level', theta='Condition', line_close=True,
                                title="Your Mental Health Footprint", range_r=[0, 100])
            fig.update_traces(fill='toself')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --- FINAL DECISION & RECOMMENDATIONS ---
            if healthy_count == 3:
                st.balloons()
                st.success("🎉 **Diagnosis: Mentally Healthy**")
                st.markdown("Your responses indicate a balanced mental state. Keep maintaining your positive lifestyle!")
                
                with st.expander("🌟 Tips to Maintain Wellness"):
                    for tip in get_recommendations("Healthy"):
                        st.markdown(tip)
            else:
                dominant = max(risk_scores, key=lambda x: x[1])
                st.error(f"🚨 **Primary Area of Concern:** {dominant[0]}")
                st.write(f"The AI model has detected patterns consistent with **{dominant[0]}**.")
                
                st.subheader(f"💡 Recommended Actions for {dominant[0]}")
                for tip in get_recommendations(dominant[0]):
                    st.info(tip)
                
                st.warning("⚠️ *Disclaimer: This AI tool is for screening purposes only and is not a substitute for professional medical advice.*")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Feature column count mismatch! Please re-download feature_columns.pkl.")
