import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
import plotly.graph_objects as go
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Student Mental Health Assessment AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .footer {text-align:center; padding:20px; font-size:12px; color:#666; border-top:1px solid #ddd; margin-top: 50px;}
    .emergency-box {background-color: #ffebee; border: 2px solid #ef5350; padding: 15px; border-radius: 10px; color: #c62828 !important; margin: 14px 0;}
    .suggestion-box {background-color: #f1f8e9; border-left: 5px solid #8bc34a; padding: 15px; border-radius: 5px; color: #33691e;}
    .suggestion-severe {background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; border-radius: 5px; color: #e65100;}
    .main-header {text-align: center; color: #4F8BF9; margin-bottom: 10px;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. LOAD RESOURCES
# -----------------------------
@st.cache_resource
def load_resources():
    try:
        # NOTE: Ensure filenames are renamed in your folder to match these:
        model = joblib.load("mental_health_hybrid_model.pkl") 
        features = joblib.load("feature_columns.pkl")
        encoders = joblib.load("label_encoders.pkl")
        return model, features, encoders
    except Exception as e:
        return None, None, None

model, feature_cols, encoders = load_resources()

# -----------------------------
# 3. HELPER FUNCTIONS (VISUALIZATION)
# -----------------------------
def get_severity_score(text_label):
    # Convert text label to numeric (0-4) for Gauge
    text_label = str(text_label).lower()
    if "normal" in text_label or "minimal" in text_label or "no " in text_label: return 0
    if "mild" in text_label: return 1
    if "moderate" in text_label and "severe" not in text_label: return 2
    if "moderately severe" in text_label or "mod. severe" in text_label: return 3
    if "severe" in text_label or "extremely" in text_label: return 4
    return 0

def create_gauge_chart(title, score):
    colors = ['#66bb6a', '#ffa726', '#ef5350'] # Green, Orange, Red
    color = colors[0] if score <= 1 else (colors[1] if score <= 2 else colors[2])
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, 4], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 1.5], 'color': '#e8f5e9'},
                {'range': [1.5, 2.5], 'color': '#fff3e0'},
                {'range': [2.5, 4], 'color': '#ffebee'}
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_radar_chart(answers_list):
    # Map questions to categories for X-AI View
    if len(answers_list) < 26: return None
    
    # Heuristic Grouping based on typical questionnaire structure
    # Adjust indices if your raw dataset columns are different
    academic_score = np.mean(answers_list[0:10]) # Q1-Q10
    anxiety_score = np.mean(answers_list[10:17]) # Q11-Q17 (GAD-7 approx)
    depression_score = np.mean(answers_list[17:26]) # Q18-Q26 (PHQ-9 approx)
    
    # Q20 is often sleep related in PHQ-9, mapping it here
    sleep_score = answers_list[19] if len(answers_list) > 19 else 0
    
    categories = ['Academic Pressure', 'Anxiety', 'Depression', 'Sleep Issues', 'General Distress']
    values = [
        academic_score, 
        anxiety_score, 
        depression_score, 
        sleep_score, 
        np.mean(answers_list)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='User Profile'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title="Personalized Risk Factor Analysis",
        height=350,
        margin=dict(t=40, b=20)
    )
    return fig

# -----------------------------
# 4. LANGUAGE DICTIONARY
# -----------------------------
translations = {
    "English": {
        "title": "Student Mental Health AI Assessment",
        "subtitle": "Powered by Hybrid Ensemble Learning | Thesis Research",
        "personal_info": "Personal Information",
        "questions": "Assessment Questions",
        "analyze_btn": "🔍 Analyze Mental Health",
        "results_title": "📊 Assessment Results",
        "disclaimer": "DISCLAIMER: This is an AI-based tool for educational/research purposes only.",
        "healthy_msg": "Great! Your results indicate a healthy mental state.",
        "suggestions": "Recommended Actions",
        "download": "Download Report (Text)",
        "tabs": ["📝 Assessment", "🔍 X-AI Analysis", "🎓 Thesis Artifacts"],
        "helpline_title": "🆘 Emergency Helplines & Resources"
    },
    "Bangla": {
        "title": "শিক্ষার্থীদের মানসিক স্বাস্থ্য মূল্যায়ন এআই",
        "subtitle": "হাইব্রিড এনসেম্বল লার্নিং দ্বারা চালিত | গবেষণাপত্র",
        "personal_info": "ব্যক্তিগত তথ্য",
        "questions": "মূল্যায়ন প্রশ্নাবলী",
        "analyze_btn": "🔍 ফলাফল বিশ্লেষণ করুন",
        "results_title": "📊 ফলাফল",
        "disclaimer": "সতর্কতা: এটি একটি এআই টুল যা শুধুমাত্র গবেষণার কাজে ব্যবহৃত।",
        "healthy_msg": "চমৎকার! আপনার মানসিক অবস্থা সুস্থ দেখাচ্ছে।",
        "suggestions": "পরামর্শসমূহ",
        "download": "রিপোর্ট ডাউনলোড করুন",
        "tabs": ["📝 মূল্যায়ন", "🔍 বিশ্লেষণ (X-AI)", "🎓 গবেষণার তথ্য"],
        "helpline_title": "🆘 জরুরি সেবা ও হেল্পলাইন"
    }
}

# -----------------------------
# 5. SIDEBAR
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=80)
    lang = st.radio("Language / ভাষা", ["English", "Bangla"])
    t = translations[lang]
    
    st.markdown("---")
    st.info("ℹ️ **About**\n\nThis system uses **Machine Learning** to assess Anxiety, Stress, and Depression levels based on academic and lifestyle factors.")
    st.warning(f"⚠️ **{t['disclaimer']}**")

# -----------------------------
# 6. MAIN APP LOGIC
# -----------------------------
# --- FEATURE 3: BANNER IMAGE (Optional) ---
try:
    # If you have a banner.jpg, put it in the same folder
    st.image("banner.jpg", use_column_width=True) 
except:
    pass # If no image, skip

st.markdown(f"<h1 class='main-header'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;'>{t['subtitle']}</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model files not found! Please upload .pkl files (renamed correctly).")
    st.stop()

# TABS SYSTEM
tab1, tab2, tab3 = st.tabs(t['tabs'])

# --- TAB 1: ASSESSMENT (FROM APP V3 + NEW IMPROVEMENTS) ---
with tab1:
    with st.form("assessment_form"):
        st.subheader(f"1. {t['personal_info']}")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age / বয়স", 18, 30, 22)
            gender = st.selectbox("Gender / লিঙ্গ", ["Male", "Female", "Other"])
            uni = st.text_input("University / বিশ্ববিদ্যালয়", "DIU")
            dept = st.text_input("Department / বিভাগ", "CSE")
        with col2:
            year = st.selectbox("Year / বর্ষ", ["1st", "2nd", "3rd", "4th", "Masters"])
            cgpa = st.number_input("CGPA", 0.0, 4.0, 3.5, step=0.01)
            scholarship = st.selectbox("Scholarship/Waiver? / স্কলারশিপ?", ["Yes", "No"])
        
        st.markdown("---")
        st.subheader(f"2. {t['questions']} (Scale: 1-5)")
        st.caption("1 = Never (কখনও না), 5 = Always (সবসময়)")
        
        # Dynamic Questions Loop
        answers = []
        question_cols = feature_cols[7:] # Skip demographics
        
        # Create 2 columns for questions to save space
        q_cols = st.columns(2)
        
        for i, col_name in enumerate(question_cols):
            with q_cols[i % 2]:
                clean_q = re.sub(r'^\d+\.\s*', '', col_name) 
                val = st.slider(f"Q{i+1}: {clean_q[:60]}...", 1, 5, 2, help=clean_q)
                answers.append(val)
        
        submitted = st.form_submit_button(t['analyze_btn'], type="primary")

    if submitted:
        # Prepare Data
        input_data = [age, gender, uni, dept, year, cgpa, scholarship] + answers
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        
        # Predict
        try:
            preds = model.predict(input_df)[0] # Multi-output prediction
            
            results = {}
            numeric_scores = {}
            target_names = ["Anxiety Label", "Stress Label", "Depression Label"]
            
            for i, target in enumerate(target_names):
                label_pred = preds[i]
                label_text = encoders[target].inverse_transform([label_pred])[0]
                results[target] = label_text
                numeric_scores[target] = get_severity_score(label_text)
            
            # Store in session state
            st.session_state['results'] = results
            st.session_state['scores'] = numeric_scores
            st.session_state['answers'] = answers
            st.session_state['submitted'] = True

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # Display Results if submitted
    if st.session_state.get('submitted'):
        res = st.session_state['results']
        scores = st.session_state['scores']
        
        st.markdown("---")
        st.subheader(t['results_title'])
        
        # 1. GAUGE CHARTS (VISUAL)
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.plotly_chart(create_gauge_chart("Anxiety", scores['Anxiety Label']), use_container_width=True)
            st.caption(f"Status: **{res['Anxiety Label']}**")
        with g_col2:
            st.plotly_chart(create_gauge_chart("Stress", scores['Stress Label']), use_container_width=True)
            st.caption(f"Status: **{res['Stress Label']}**")
        with g_col3:
            st.plotly_chart(create_gauge_chart("Depression", scores['Depression Label']), use_container_width=True)
            st.caption(f"Status: **{res['Depression Label']}**")

        # 2. SUGGESTIONS
        st.markdown("---")
        st.subheader(t['suggestions'])
        
        def get_suggestions(category, level, language):
            is_bangla = language == "Bangla"
            if "Severe" in level or "Moderate" in level:
                if "Anxiety" in category:
                    return ["Practice deep breathing (4-7-8).", "Limit caffeine.", "Consult a counselor."] if not is_bangla else ["গভীর শ্বাস-প্রশ্বাসের ব্যায়াম করুন।", "ক্যাফেইন কমান।", "কাউন্সিলরের পরামর্শ নিন।"]
                elif "Stress" in category:
                    return ["Make a strict study routine.", "Take short breaks.", "Exercise daily."] if not is_bangla else ["রুটিন মেনে চলুন।", "ছোট বিরতি নিন।", "ব্যায়াম করুন।"]
                elif "Depression" in category:
                    return ["Talk to friends/family.", "Sleep 7-8 hours.", "Seek therapy."] if not is_bangla else ["পরিবারের সাথে কথা বলুন।", "পর্যাপ্ত ঘুমান।", "থেরাপি নিন।"]
            else:
                return ["Keep up the healthy routine.", "Stay positive."] if not is_bangla else ["বর্তমান রুটিন বজায় রাখুন।", "ইতিবাচক থাকুন।"]
            return []

        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for idx, target in enumerate(["Anxiety Label", "Stress Label", "Depression Label"]):
            val = res[target]
            cat_name = target.split()[0]
            tips = get_suggestions(cat_name, val, lang)
            with cols[idx]:
                style = "suggestion-severe" if "Severe" in val or "Moderate" in val else "suggestion-box"
                st.markdown(f"""
                <div class='{style}'>
                    <b>{cat_name}: {val}</b><br>
                    <small>{'- ' + '<br>- '.join(tips)}</small>
                </div>
                """, unsafe_allow_html=True)

        # --- FEATURE 1: HELPLINE (Static UI) ---
        with st.expander(t['helpline_title']):
            st.markdown("""
            - **Kaan Pete Roi:** 01779554391 (Mental Health Support)
            - **National Emergency:** 999
            - **DIU Counseling Unit:** counseling@diu.edu.bd
            - **Location:** Daffodil Smart City, Ashulia.
            """)

        # --- FEATURE 2: FEEDBACK BUTTON ---
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            # 3. DOWNLOAD BUTTON
            report_text = f"Mental Health Report\nDate: {datetime.now()}\n\n"
            for k, v in res.items(): report_text += f"{k}: {v}\n"
            st.download_button(t['download'], data=report_text, file_name="report.txt", mime="text/plain")
        
        with col_btn2:
            if st.button("📢 Report Bug / Feedback"):
                st.toast("Thank you for your feedback! We will review it.")

# --- TAB 2: X-AI ANALYSIS ---
with tab2:
    st.header(t['tabs'][1])
    if st.session_state.get('submitted'):
        st.markdown("### Personalized Risk Factor Analysis")
        st.info("This chart visualizes which areas of your life (Academic, Sleep, etc.) are contributing most to your stress.")
        radar_fig = create_radar_chart(st.session_state['answers'])
        if radar_fig: st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Please submit the assessment in the first tab to see the analysis.")

# --- TAB 3: THESIS ARTIFACTS ---
with tab3:
    st.header(t['tabs'][2])
    st.markdown("""
    ### 🔬 Research Methodology
    This system is built upon a **Hybrid Ensemble Classifier** developed during the undergraduate thesis research.
    **Key Highlights:**
    - **Models Used:** Logistic Regression + CatBoost (Soft Voting Ensemble)
    - **Accuracy:** ~98% (Validated via 5-Fold Cross Validation)
    """)
    st.markdown("*(Upload confusion matrix images in your folder to display them here)*")
    # st.image("Thesis_Final_Artifacts/CM_Anxiety.png") 

# Footer
st.markdown("<div class='footer'>© 2024 Thesis Project | Department of CSE | DIU</div>", unsafe_allow_html=True)
