import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. PAGE CONFIGURATION & CUSTOM STYLING ---
st.set_page_config(
    page_title="Student Mental Health AI", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f8f9fa;
        border-radius: 5px; gap: 1px; padding: 10px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd; border-bottom: 3px solid #2196F3; color: #0d47a1;
    }
    /* Expander Styling */
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: 600; }
    /* Question Highlight */
    .question-box {
        background-color: #ffffff; padding: 15px; border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; border-left: 4px solid #2196F3;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #fff; border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px;
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
            "🌬️ **Deep Breathing:** Try the 4-7-8 technique (Inhale 4s, Hold 7s, Exhale 8s).",
            "🧘 **Grounding:** Name 5 things you see, 4 you feel, 3 you hear.",
            "☕ **Limit Caffeine:** Reduce coffee/tea intake."
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

# --- 3. HEADER & ARCHITECTURE ---
col1, col2 = st.columns([8, 2])
with col1:
    st.title("🧠 AI-Powered Student Mental Health Assessment")
    st.markdown("##### An Intelligent Screening System developed by **Team Dual Core**")
with col2:
    if st.button("🔄 Reset System", type="primary"):
        reset_app()

with st.expander("ℹ️ How It Works (System Architecture)"):
    st.info("""
    **1. Input Layer:** User provides demographics & behavioral inputs (33 Features).  
    **2. Hybrid AI Engine:** Ensemble Model (Random Forest + Gradient Boosting) analyzes patterns.  
    **3. Diagnostic Output:** Probabilistic prediction of Anxiety, Stress, and Depression.
    """)
    # Diagram Placeholder
    # 

# --- 4. SIDEBAR PROFILE & HELPLINE ---
st.sidebar.header("📝 Student Profile")
st.sidebar.success("""
**Developed by: Team Dual Core** 👨‍💻 Ovi Sarker  
👨‍💻 BM Sabbir Hossen Riad  
**Dept. of CSE** Daffodil International University 🎓
""")

# --- NEW FEATURE: HELPLINE ---
with st.sidebar.expander("🆘 Emergency Helpline (BD)", expanded=True):
    st.markdown("""
    📞 **Kaan Pete Roi:** 01779554391  
    📞 **Moner Bondhu:** 01779632588  
    🚑 **National Emergency:** 999
    """)

st.sidebar.markdown("---")

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

# --- 5. ELEGANT SERIAL QUESTIONNAIRE ---
st.markdown("---")
st.subheader("📋 Behavioral Self-Assessment")
st.caption("Please slide the scale to match how you felt over the last 2 weeks.")

# Slider Logic
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

# Tabs for Serial Flow
tab1, tab2, tab3 = st.tabs(["📚 Part 1: Academic & Life", "⚡ Part 2: Anxiety & Stress", "🌧️ Part 3: Mood & Physical"])
answers_map = {}

def render_questions_with_slider(tab, start_idx, end_idx):
    with tab:
        for i in range(start_idx, end_idx):
            q_text = q_labels[i]
            key_name = f"q_{i}_{st.session_state.reset}"
            # Elegant Slider
            val = st.select_slider(
                label=f"**{q_text}**", 
                options=slider_options, 
                value=slider_options[0], 
                key=key_name
            )
            answers_map[i] = options_map[val]

render_questions_with_slider(tab1, 0, 10)
render_questions_with_slider(tab2, 10, 17)
render_questions_with_slider(tab3, 17, 26)

# Reconstruct ordered answers
final_answers = [answers_map[i] for i in range(26)]

if st.session_state.reset:
    st.session_state.reset = False

# --- 6. PREDICTION ENGINE ---
st.markdown("---")
col_cen1, col_cen2, col_cen3 = st.columns([1, 2, 1])
with col_cen2:
    analyze_btn = st.button("🚀 Analyze Mental Health Status", type="primary", use_container_width=True)

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
            with st.spinner("AI Model is analyzing patterns..."):
                probs = model.predict_proba(input_df)
            
            st.success("✅ Analysis Complete")
            st.subheader("📊 Diagnostic Report")
            
            result_cols = st.columns(3)
            conditions = ['Anxiety', 'Stress', 'Depression']
            risk_scores = []
            healthy_count = 0
            
            # Text Report Content
            report_text = f"--- MENTAL HEALTH ASSESSMENT REPORT ---\n"
            report_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_text += f"Student ID Ref: (Anonymous)\n"
            report_text += f"Profile: {age_input}, {gender}, {dept}, Year: {year}\n"
            report_text += "---------------------------------------\n\n"
            
            for i, cond in enumerate(conditions):
                prob_arr = probs[i][0]
                best_idx = np.argmax(prob_arr)
                label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                confidence = prob_arr[best_idx] * 100
                
                # Smart Label Mapping
                display_label = label
                if label == "Minimal Anxiety": display_label = "No Anxiety / Healthy"
                if label == "Low Stress": display_label = "No Stress / Healthy"
                if label in ["No Depression", "Minimal Depression"]: display_label = "No Depression / Healthy"

                is_healthy = any(safe in label for safe in ["Minimal", "Low", "None", "No Depression"])
                
                # Add to text report
                report_text += f"{cond}: {display_label} (Confidence: {confidence:.1f}%)\n"
                
                with result_cols[i]:
                    st.markdown(f"#### {cond}")
                    if is_healthy:
                        st.success(f"**{display_label}**")
                        st.progress(0)
                        st.caption(f"Risk: Low ({confidence:.1f}%)")
                        risk_scores.append((cond, 0))
                        healthy_count += 1
                    else:
                        st.error(f"**{display_label}**")
                        st.progress(int(confidence))
                        st.caption(f"Risk: High ({confidence:.1f}%)")
                        risk_scores.append((cond, confidence))

            st.markdown("---")
            
            # --- VISUALIZATION & RECOMMENDATIONS ---
            col_v1, col_v2 = st.columns([1, 1])
            
            with col_v1:
                st.subheader("📈 Emotional Footprint")
                viz_scores = [score if score > 0 else 5 for _, score in risk_scores]
                df_chart = pd.DataFrame({'Condition': conditions, 'Risk Level': viz_scores})
                fig = px.line_polar(df_chart, r='Risk Level', theta='Condition', line_close=True, range_r=[0, 100], template="plotly_white")
                fig.update_traces(fill='toself', line_color='#FF4B4B')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_v2:
                st.subheader("💡 AI Recommendations")
                if healthy_count == 3:
                    st.balloons()
                    st.success("🎉 **Great News!** No significant issues detected.")
                    report_text += "\n[Recommendation]: Maintain current healthy lifestyle."
                    for tip in get_recommendations("Healthy"):
                        st.info(tip)
                else:
                    dominant = max(risk_scores, key=lambda x: x[1])
                    st.warning(f"🚨 **Primary Concern: {dominant[0]}**")
                    report_text += f"\n[Primary Concern]: {dominant[0]}\n[Suggested Actions]:\n"
                    st.markdown(f"The AI detected patterns consistent with **{dominant[0]}**.")
                    for tip in get_recommendations(dominant[0]):
                        st.warning(tip)
                        report_text += f"- {tip.replace('**', '')}\n"

            # --- NEW FEATURE: DOWNLOAD REPORT ---
            st.markdown("---")
            report_text += "\n---------------------------------------\n"
            report_text += "DISCLAIMER: This is an AI-generated screening result, not a medical diagnosis."
            
            st.download_button(
                label="📥 Download Full Report (Text)",
                data=report_text,
                file_name=f"Mental_Health_Report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                type="primary"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Feature column count mismatch! Please re-download .pkl files.")

# --- 7. FOOTER DISCLAIMER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; color: #856404; font-size: 13px; text-align: center;">
    <strong>⚠️ Medical Disclaimer:</strong> <br>
    This application is an AI-based research prototype developed for academic purposes by <strong>Team Dual Core</strong>. 
    The results provided are probabilistic predictions based on data patterns and <strong>do not constitute a professional medical diagnosis</strong>. 
    If you are feeling overwhelmed, please consult a certified mental health professional immediately.
</div>
""", unsafe_allow_html=True)
