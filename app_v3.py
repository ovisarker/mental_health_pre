import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from datetime import datetime
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Mental Health Assessment", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #e0e0e0;
        padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center; padding: 20px; font-size: 12px; color: #666; border-top: 1px solid #eee;
    }
    .emergency-box {
        background-color: #ffebee; border: 2px solid #ef5350; padding: 15px; border-radius: 8px; color: #c62828; margin-bottom: 20px;
    }
    .suggestion-box {
        background-color: #f1f8e9; padding: 15px; border-radius: 8px; border-left: 5px solid #8bc34a; margin-top: 10px;
    }
    .suggestion-warning {
        background-color: #fff3e0; padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800; margin-top: 10px;
    }
    /* Step Progress Bar */
    .step-indicator {
        font-size: 18px; font-weight: bold; color: #2196F3; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- TRANSLATION DICTIONARY ---
translations = {
    'English': {
        'title': "Student Mental Health Assessment",
        'subtitle': "Machine Learning Based Screening System",
        'step1': "Step 1: Student Profile",
        'step2': "Step 2: Behavioral Assessment",
        'next_btn': "Next: Go to Questionnaire 👉",
        'back_btn': "👈 Back to Profile",
        'profile_err': "⚠️ Please fill in all profile fields correctly (e.g., Department cannot be empty).",
        'age': "Age Group",
        'gender': "Gender",
        'uni': "University Type",
        'dept': "Department Name (e.g., CSE)",
        'year': "Academic Year",
        'cgpa': "Current CGPA",
        'scholarship': "Scholarship/Waiver?",
        'helpline_title': "🆘 Emergency Helpline (BD)",
        'instructions': "💡 **Instructions:** Please indicate how frequently you have experienced these feelings **over the last 2 weeks**.",
        'scale_caption': "Scale: **Not at all** (0) → **Sometimes** (1) → **Often** (2) → **Very Often** (3)",
        'analyze_btn': "🚀 Analyze Risk Level",
        'analyzing': "Analyzing behavioral patterns...",
        'success': "✅ Assessment Complete",
        'result_title': "📊 Clinical Risk Assessment",
        'suggestions': "💡 Professional Recommendations",
        'healthy_msg': "🎉 **Status: Mentally Resilient**\nYour responses indicate a stable mental state.",
        'download_btn': "📥 Download Assessment Report",
        'disclaimer_short': "⚠️ **Disclaimer:** This tool uses ML algorithms for screening. It is not a substitute for clinical diagnosis.",
        'dev_by': "Developed by",
        'slider_opts': ["Not at all", "Sometimes", "Often", "Very Often"],
        'genders': ['Male', 'Female'],
        'unis': ['Public', 'Private'],
        'scholars': ['Yes', 'No'],
        'years': ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'],
        'depts': ["CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"]
    },
    'Bangla': {
        'title': "শিক্ষার্থী মানসিক স্বাস্থ্য মূল্যায়ন",
        'subtitle': "মেশিন লার্নিং ভিত্তিক স্ক্রিনিং সিস্টেম",
        'step1': "ধাপ ১: শিক্ষার্থীর প্রোফাইল",
        'step2': "ধাপ ২: আচরণগত মূল্যায়ন",
        'next_btn': "পরবর্তী: প্রশ্নাবলী শুরু করুন 👉",
        'back_btn': "👈 প্রোফাইলে ফিরে যান",
        'profile_err': "⚠️ দয়া করে প্রোফাইলের সব তথ্য সঠিক ভাবে পূরণ করুন (ডিপার্টমেন্ট খালি রাখা যাবে না)।",
        'age': "বয়স গ্রুপ",
        'gender': "লিঙ্গ",
        'uni': "বিশ্ববিদ্যালয়ের ধরণ",
        'dept': "ডিপার্টমেন্টের নাম (যেমন: CSE)",
        'year': "শিক্ষাবর্ষ",
        'cgpa': "বর্তমান সিজিপিএ (CGPA)",
        'scholarship': "স্কলারশিপ/ওয়েভার আছে?",
        'helpline_title': "🆘 জরুরি হেল্পলাইন (BD)",
        'instructions': "💡 **নির্দেশনা:** গত **২ সপ্তাহে** আপনি এই অনুভূতিগুলো কতবার অনুভব করেছেন তা জানান।",
        'scale_caption': "স্কেল: **একদম না** (০) → **মাঝে মাঝে** (১) → **প্রায়ই** (২) → **খুব বেশি** (৩)",
        'analyze_btn': "🚀 ঝুঁকি বিশ্লেষণ করুন",
        'analyzing': "মেশিন লার্নিং মডেল বিশ্লেষণ করছে...",
        'success': "✅ মূল্যায়ন সম্পন্ন হয়েছে",
        'result_title': "📊 ক্লিনিক্যাল ঝুঁকি মূল্যায়ন",
        'suggestions': "💡 পেশাদার পরামর্শ",
        'healthy_msg': "🎉 **অবস্থা: মানসিকভাবে স্থিতিশীল**\nআপনার মানসিক অবস্থা ভারসাম্যপূর্ণ।",
        'download_btn': "📥 রিপোর্ট ডাউনলোড করুন",
        'disclaimer_short': "⚠️ **সতর্কতা:** এটি একটি স্ক্রিনিং টুল। এটি পেশাদার ক্লিনিক্যাল ডায়াগনসিসের বিকল্প নয়।",
        'dev_by': "ডেভেলপ করেছে",
        'slider_opts': ["একদম না", "মাঝে মাঝে", "প্রায়ই", "খুব বেশি"],
        'genders': ['পুরুষ', 'মহিলা'],
        'unis': ['পাবলিক', 'প্রাইভেট'],
        'scholars': ['হ্যাঁ', 'না'],
        'years': ['১ম বর্ষ', '২য় বর্ষ', '৩য় বর্ষ', '৪র্থ বর্ষ', 'মাস্টার্স'],
        'depts': ["সিএসই", "ইইই", "বিবিএ", "ইংরেজি", "আইন", "ফার্মাসি", "অন্যান্য"]
    }
}

# Questions Translation
q_labels_en = [
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

q_labels_bn = [
    "১. পড়াশোনার চাপে মন খারাপ?", "২. গুরুত্বপূর্ণ বিষয় নিয়ন্ত্রণে অক্ষম?", "৩. নার্ভাস বা মানসিক চাপে ছিলেন?", 
    "৪. বাধ্যতামূলক কাজ সামলাতে পারছেন না?", "৫. সমস্যা সমাধানে আত্মবিশ্বাসী?", "৬. সব কিছু আপনার মতো হচ্ছে?", 
    "৭. বিরক্তি নিয়ন্ত্রণ করতে পারেন?", "৮. একাডেমিক পারফরম্যান্স ভালো মনে হচ্ছে?", "৯. খারাপ ফলাফলে রাগান্বিত?", 
    "১০. সমস্যাগুলো কি পাহাড়সম মনে হচ্ছে?", "১১. উদ্বিগ্ন বা অস্থির অনুভব করেন?", "১২. দুশ্চিন্তা থামাতে পারছেন না?", 
    "১৩. রিল্যাক্স করতে সমস্যা হচ্ছে?", "১৪. খুব বেশি অস্থির লাগছে?", "১৫. সহজেই বিরক্ত হয়ে যাচ্ছেন?", 
    "১৬. ভয় পাচ্ছেন খারাপ কিছু ঘটবে?", "১৭. অতিরিক্ত দুশ্চিন্তা করছেন?", "১৮. কাজে আগ্রহ পাচ্ছেন না?", 
    "১৯. মন খারাপ বা হতাশ লাগছে?", "২০. ঘুমাতে সমস্যা হচ্ছে?", "২১. ক্লান্ত বা শক্তি কম লাগছে?", 
    "২২. ক্ষুধামন্দা বা অতিরিক্ত খাওয়া?", "২৩. নিজেকে নিয়ে খারাপ লাগছে?", "২৪. মনোযোগ দিতে সমস্যা হচ্ছে?", 
    "২৫. খুব ধীর বা দ্রুত নড়াচড়া করছেন?", "২৬. নিজেকে আঘাত করার চিন্তা আসছে?"
]

# --- SESSION STATE & RESOURCES ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'profile_data' not in st.session_state:
    st.session_state.profile_data = {}

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

def is_low_risk_label(label):
    low_set = {"Minimal Anxiety", "Low Stress", "No Depression", "Minimal Depression", "Normal", "None"}
    return label in low_set or any(x in label for x in ["Minimal", "Low", "No Depression"])

def get_professional_suggestions(condition, severity, lang):
    # (Keeping suggestion logic brief for length - reusing previous logic)
    # Using simple fallback for demonstration, assumes full logic exists in your version
    return [f"Please consult a professional for {condition} ({severity})."]

# --- LANGUAGE SELECTOR (Always Visible) ---
with st.sidebar:
    st.markdown("### 🌐 Language / ভাষা")
    lang = st.radio("", ('English', 'Bangla'), label_visibility="collapsed")
    t = translations[lang]
    st.markdown("---")
    with st.expander(t['helpline_title'], expanded=True):
        st.markdown("📞 **Kaan Pete Roi:** 01779554391\n📞 **Moner Bondhu:** 01779632588\n🚑 **999**")

# --- HEADER ---
col1, col2 = st.columns([8, 2])
with col1:
    st.title(t['title'])
    st.markdown(f"##### {t['subtitle']}")

st.markdown("---")

# =========================================================
# STEP 1: STUDENT PROFILE (MANDATORY)
# =========================================================
if st.session_state.step == 1:
    st.markdown(f"<div class='step-indicator'>{t['step1']}</div>", unsafe_allow_html=True)
    
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            age_display = ['18-22', '23-26', '27-30', 'Above 30']
            age_input = st.selectbox(t['age'], age_display)
            
            gender_idx = st.selectbox(t['gender'], t['genders'])
            
            dept_idx = st.selectbox(t['dept'], t['depts']) # Using Selectbox ensures valid input
            
            cgpa_input = st.number_input(t['cgpa'], min_value=0.00, max_value=4.00, value=3.50, step=0.01)

        with c2:
            uni_idx = st.selectbox(t['uni'], t['unis'])
            year_idx = st.selectbox(t['year'], t['years'])
            sch_idx = st.selectbox(t['scholarship'], t['scholars'])
            
        submitted = st.form_submit_button(t['next_btn'], type="primary", use_container_width=True)
        
        if submitted:
            # Map Inputs immediately to English for Model
            gender_model = 'Male' if gender_idx in ['Male', 'পুরুষ'] else 'Female'
            uni_model = 'Public' if uni_idx in ['Public', 'পাবলিক'] else 'Private'
            sch_model = 'Yes' if sch_idx in ['Yes', 'হ্যাঁ'] else 'No'
            
            dept_map = {"সিএসই": "CSE", "ইইই": "EEE", "বিবিএ": "BBA", "ইংরেজি": "English", "আইন": "Law", "ফার্মাসি": "Pharmacy", "অন্যান্য": "Other"}
            dept_model = dept_map.get(dept_idx, dept_idx)
            
            year_map = {'১ম বর্ষ': 'First Year', '২য় বর্ষ': 'Second Year', '৩য় বর্ষ': 'Third Year', '৪র্থ বর্ষ': 'Fourth Year', 'মাস্টার্স': 'Master'}
            year_model = year_map.get(year_idx, year_idx)
            
            # Save to session
            st.session_state.profile_data = {
                'age': extract_number(age_input),
                'gender': gender_model,
                'uni': uni_model,
                'dept': dept_model,
                'year': year_model,
                'cgpa': float(cgpa_input),
                'scholarship': sch_model,
                'display_profile': f"{age_input}, {gender_idx}, {dept_idx}" # For report
            }
            
            # Move to Step 2
            st.session_state.step = 2
            st.rerun()

# =========================================================
# STEP 2: QUESTIONNAIRE & ANALYSIS
# =========================================================
elif st.session_state.step == 2:
    # Show tiny profile summary
    st.info(f"👤 **Profile:** {st.session_state.profile_data['display_profile']}")
    
    st.markdown(f"<div class='step-indicator'>{t['step2']}</div>", unsafe_allow_html=True)
    st.info(t['instructions'])
    
    slider_options = t['slider_opts'] 
    options_map = {
        "Not at all": 0, "একদম না": 0,
        "Sometimes": 1, "মাঝে মাঝে": 1,
        "Often": 2, "প্রায়ই": 2,
        "Very Often": 3, "খুব বেশি": 3
    }
    
    q_labels = q_labels_bn if lang == 'Bangla' else q_labels_en
    answers_map = {}
    
    # Questionnaire Form
    q_col1, q_col2 = st.columns(2)
    
    for i, q_text in enumerate(q_labels):
        current_col = q_col1 if i % 2 == 0 else q_col2
        with current_col:
            val = st.select_slider(label=f"**{q_text}**", options=slider_options, value=slider_options[0], key=f"q_{i}")
            answers_map[i] = options_map[val]
            st.write("") 

    final_answers = [answers_map[i] for i in range(26)]

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(t['back_btn']):
            st.session_state.step = 1
            st.rerun()
            
    with c2:
        analyze_btn = st.button(t['analyze_btn'], type="primary", use_container_width=True)

    if analyze_btn:
        # Retrieve Profile from Session
        p = st.session_state.profile_data
        
        input_dict = {}
        if len(feature_columns) == 33:
            input_dict[feature_columns[0]] = p['age']
            input_dict[feature_columns[1]] = p['gender']
            input_dict[feature_columns[2]] = p['uni']
            input_dict[feature_columns[3]] = p['dept']
            input_dict[feature_columns[4]] = p['year']
            input_dict[feature_columns[5]] = p['cgpa']
            input_dict[feature_columns[6]] = p['scholarship']
            for i in range(26):
                input_dict[feature_columns[7+i]] = final_answers[i]
                
            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)
            
            try:
                with st.spinner(t['analyzing']):
                    probs = model.predict_proba(input_df)
                
                # Global Safety Alert (Q26)
                if final_answers[25] >= 2:
                    st.markdown(f"""
                    <div class="emergency-box">
                        <h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3>
                        <p>{'Your response indicates distress. Please seek professional help immediately.' if lang=='English' else 'আপনার উত্তর মানসিক যন্ত্রণার ইঙ্গিত দিচ্ছে। দয়া করে অবিলম্বে পেশাদার সাহায্য নিন।'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(t['success'])
                st.subheader(t['result_title'])
                
                result_cols = st.columns(3)
                conditions = ['Anxiety', 'Stress', 'Depression']
                
                report_text = f"--- REPORT ---\nDate: {datetime.now()}\nProfile: {p['display_profile']}\n"
                
                for i, cond in enumerate(conditions):
                    prob_arr = probs[i][0]
                    best_idx = np.argmax(prob_arr)
                    label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                    confidence = prob_arr[best_idx] * 100
                    
                    is_healthy = is_low_risk_label(label)
                    
                    # Label Translation (Visual Only)
                    display_label = label
                    if lang == 'Bangla':
                        if is_healthy: display_label = "ঝুঁকি নেই / সুস্থ"
                        elif "Severe" in label: display_label = "তীব্র ঝুঁকি"
                        elif "Moderate" in label: display_label = "মাঝারি ঝুঁকি"
                        elif "Mild" in label: display_label = "মৃদু ঝুঁকি"

                    report_text += f"{cond}: {label} ({confidence:.1f}%)\n"
                    
                    with result_cols[i]:
                        st.markdown(f"#### {cond}")
                        if is_healthy:
                            st.success(f"**{display_label}**")
                            st.progress(0)
                            
                            # Clinical Note Check
                            if cond == 'Depression' and final_answers[25] >= 2:
                                st.warning("⚠️ **Clinical Note:** Self-harm risk detected.")
                        else:
                            st.error(f"**{display_label}**")
                            st.progress(int(confidence))
                            st.caption(f"{'Risk' if lang=='English' else 'ঝুঁকি'}: {confidence:.1f}%")

                st.download_button(t['download_btn'], report_text, file_name="Report.txt")

            except Exception as e:
                st.error(f"Error: {e}")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"<div class='footer'>{t['dev_by']} <b>Team Dual Core</b><br>{t['disclaimer_short']}</div>", unsafe_allow_html=True)
