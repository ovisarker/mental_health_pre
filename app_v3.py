import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Mental Health Assessment", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Footer & Cards
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f3f6;
        color: #333;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #e0e0e0;
        z-index: 100;
    }
    .footer-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        font-family: sans-serif;
    }
    .dev-section {
        text-align: center;
        line-height: 1.4;
        font-size: 14px;
        color: #555;
    }
    .dev-title {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 12px;
    }
    .dev-name {
        font-weight: 600;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# --- RESET LOGIC ---
if 'reset' not in st.session_state:
    st.session_state.reset = False

def reset_app():
    st.session_state.reset = True
    st.rerun()

# --- 2. LOAD RESOURCES ---
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

# Helper Functions
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

def get_recommendations(condition):
    tips = {
        "Anxiety": [
            "🌬️ **Deep Breathing:** Try the 4-7-8 technique.",
            "🧘 **Grounding:** Name 5 things you see, 4 you feel.",
            "☕ **Limit Caffeine:** Reduce coffee/tea intake."
        ],
        "Stress": [
            "📝 **Prioritize:** Make a to-do list.",
            "🚶 **Move:** A 10-minute walk lowers cortisol.",
            "💤 **Sleep:** Ensure 7-8 hours of sleep."
        ],
        "Depression": [
            "🤝 **Connect:** Talk to a friend today.",
            "🌞 **Sunlight:** Spend 15 mins outside.",
            "📅 **Routine:** Stick to a small daily routine."
        ],
        "Healthy": [
            "🎉 **Keep going!** Resilience is strong.",
            "💧 **Stay Hydrated:** Drink enough water.",
            "📖 **Journal:** Write down 3 good things."
        ]
    }
    return tips.get(condition, [])

# --- 3. HEADER ---
col1, col2 = st.columns([8, 2])
with col1:
    st.title("Student Mental Health Assessment & Prediction")
    st.markdown("##### A Machine Learning Based Screening System")
with col2:
    if st.button("🔄 Reset Form", type="primary"):
        reset_app()

st.markdown("---")

# --- 4. SIDEBAR PROFILE ---
st.sidebar.header("📝 Student Profile")

def get_index(options, default_idx=0):
    return 0 if st.session_state.reset else default_idx

# Profile Inputs
age_input = st.sidebar.selectbox("1. Age Group", ['18-22', '23-26', '27-30', 'Above 30'], index=get_index(4))
gender = st.sidebar.selectbox("2. Gender", ['Male', 'Female'], index=get_index(2))
uni = st.sidebar.selectbox("3. University Type", ['Public', 'Private'], index=get_index(2))
dept = st.sidebar.text_input("4. Department", value="" if st.session_state.reset else "CSE")
year = st.sidebar.selectbox("5. Academic Year", ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'], index=get_index(5))
cgpa_val = 0.00 if st.session_state.reset else 3.50
cgpa_input = st.sidebar.number_input("6. Current CGPA", min_value=0.00, max_value=4.00, value=cgpa_val, step=0.01, format="%.2f")
scholarship = st.sidebar.selectbox("7. Scholarship/Waiver?", ['Yes', 'No'], index=get_index(2))

st.sidebar.markdown("---")

# --- HELPLINE (Sidebar Bottom) ---
with st.sidebar.expander("🆘 Emergency Helpline (BD)", expanded=True):
    st.markdown("""
    📞 **Kaan Pete Roi:** 01779554391  
    📞 **Moner Bondhu:** 01779632588  
    🚑 **National Emergency:** 999
    """)

# --- 5. QUESTIONNAIRE (SINGLE SECTION) ---
st.subheader("📋 Behavioral Self-Assessment")
st.info("💡 **Instructions:** Please slide the scale to indicate how frequently you have felt these emotions **over the last semester**.")
st.caption("Scale: **Not at all** (0) → **Sometimes** (1) → **Often** (2) → **Very Often** (3)")

slider_options = ["Not at all", "Sometimes", "Often", "Very Often"]
options_map = {
    "Not at all": 0, "Not at all / Never": 0,
    "Sometimes": 1, "Several days / Sometimes": 1,
    "Often": 2, "More than half the days / Often": 2,
    "Very Often": 3, "Nearly every day / Very Often": 3
}

q_labels = [
    "1. Upset due to academic affairs?", "2. Unable to control important things?", "3. Nervous and stressed?", 
    "4. Could not cope with mandatory activities?", "5. Confident about handling problems?", "6. Things going your way?", 
    "7. Able to control irritations?", "8. Felt academic performance was on top?", "9. Angered due to bad performance?", 
    "10. Difficulties piling up?", "11. Nervous, anxious or on edge?", "12. Unable to stop worrying?", 
    "13. Trouble relaxing?", "14. Being so restless?", "15. Easily annoyed or irritable?", 
    "16. Afraid something awful might happen?", "17. Worrying too much?", "18. Little interest in doing things?", 
    "19. Feeling down, depressed, or hopeless?", "20. Trouble falling/staying asleep?", "21. Feeling tired/little energy?", 
    "22. Poor appetite or overeating?", "23. Feeling bad about yourself?", "24. Trouble concentrating?", 
    "25. Moving slowly or too fast?", "26. Thoughts of hurting yourself?"
]

answers_map = {}
q_col1, q_col2 = st.columns(2)

for i, q_text in enumerate(q_labels):
    current_col = q_col1 if i % 2 == 0 else q_col2
    with current_col:
        key_name = f"q_{i}_{st.session_state.reset}"
        val = st.select_slider(label=f"**{q_text}**", options=slider_options, value=slider_options[0], key=key_name)
        answers_map[i] = options_map[val]
        st.write("") 

final_answers = [answers_map[i] for i in range(26)]

if st.session_state.reset:
    st.session_state.reset = False

# --- 6. PREDICTION ENGINE ---
st.markdown("---")
col_cen1, col_cen2, col_cen3 = st.columns([1, 2, 1])
with col_cen2:
    analyze_btn = st.button("🚀 Analyze Prediction", type="primary", use_container_width=True)

if analyze_btn:
    age_numeric = extract_number(age_input)
    cgpa_numeric = float(cgpa_input)
    
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
            input_dict[feature_columns[7+i]] = final_answers[i]
            
        input_df = pd.DataFrame([input_dict])
        
        try:
            with st.spinner("Analyzing pattern..."):
                probs = model.predict_proba(input_df)
            
            st.success("✅ Assessment Complete")
            st.subheader("📊 Prediction Results")
            
            result_cols = st.columns(3)
            conditions = ['Anxiety', 'Stress', 'Depression']
            risk_scores = []
            healthy_count = 0
            
            report_text = f"--- MENTAL HEALTH ASSESSMENT REPORT ---\n"
            report_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_text += f"Profile: {age_input}, {gender}, {dept}\n"
            report_text += "---------------------------------------\n\n"
            
            for i, cond in enumerate(conditions):
                prob_arr = probs[i][0]
                best_idx = np.argmax(prob_arr)
                label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                confidence = prob_arr[best_idx] * 100
                
                display_label = label
                if label == "Minimal Anxiety": display_label = "No Anxiety / Healthy"
                if label == "Low Stress": display_label = "No Stress / Healthy"
                if label in ["No Depression", "Minimal Depression"]: display_label = "No Depression / Healthy"

                is_healthy = any(safe in label for safe in ["Minimal", "Low", "None", "No Depression"])
                
                report_text += f"{cond}: {display_label} (Confidence: {confidence:.1f}%)\n"
                
                with result_cols[i]:
                    st.markdown(f"#### {cond}")
                    if is_healthy:
                        st.success(f"**{display_label}**")
                        st.progress(0)
                        risk_scores.append((cond, 0))
                        healthy_count += 1
                    else:
                        st.error(f"**{display_label}**")
                        st.progress(int(confidence))
                        st.caption(f"Risk Probability: {confidence:.1f}%")
                        risk_scores.append((cond, confidence))

            st.markdown("---")
            
            # --- RECOMMENDATIONS (No Chart) ---
            st.subheader("💡 Suggestions")
            if healthy_count == 3:
                st.balloons()
                st.success("🎉 **Status: Healthy**")
                st.markdown("Your input pattern suggests a balanced mental state.")
                for tip in get_recommendations("Healthy"):
                    st.info(tip)
            else:
                dominant = max(risk_scores, key=lambda x: x[1])
                st.warning(f"🚨 **Primary Concern: {dominant[0]}**")
                for tip in get_recommendations(dominant[0]):
                    st.info(tip)

            # Download Report
            st.markdown("---")
            report_text += "\n---------------------------------------\n"
            report_text += "DISCLAIMER: This result is based on ML prediction patterns and is not a clinical diagnosis."
            
            st.download_button(
                label="📥 Download Assessment Report",
                data=report_text,
                file_name=f"Assessment_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Feature column count mismatch!")

# --- 7. MODERN FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<div style="text-align: center; font-family: sans-serif; color: #555;">
    <p style="font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">
        Developed by
    </p>
    <h3 style="margin: 0; color: #2c3e50;">Team Dual Core</h3>
    <p style="margin: 5px 0 0 0; font-size: 14px;">Ovi Sarker</p>
    <p style="margin: 2px 0 0 0; font-size: 14px;">BM Sabbir Hossen Riad</p>
    <p style="margin-top: 10px; font-size: 12px; color: #7f8c8d;">
        Department of CSE • Daffodil International University
    </p>
    <br>
    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; display: inline-block; border: 1px solid #ffeeba;">
        <span style="font-size: 11px; color: #856404;">
            ⚠️ <strong>Disclaimer:</strong> This is a Machine Learning based research project. 
            Results are probabilistic and do not replace professional medical advice.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
