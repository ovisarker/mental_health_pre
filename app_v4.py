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

# --- CUSTOM CSS (FROM APP V3 + UPDATES) ---
st.markdown("""
<style>
    /* 1. General UI Elements */
    .footer {text-align:center; padding:20px; font-size:12px; color:#666; border-top:1px solid #ddd; margin-top: 50px;}
    
    .emergency-box {
        background-color: #ffebee; 
        border: 2px solid #ef5350; 
        padding: 15px; 
        border-radius: 10px; 
        color: #c62828 !important; 
        margin: 14px 0;
    }
    
    .locked-hint {
        background-color: #f8f9fa; 
        border: 1px solid #ddd; 
        padding: 14px; 
        border-radius: 10px; 
        color: #333 !important;
    }

    /* 2. SUGGESTION BOXES */
    .suggestion-box {
        background-color: #f0f7ff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #007bff; 
        margin: 10px 0; 
        color: #000000 !important;
    }
    
    .suggestion-severe {
        background-color: #fff3cd; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #ffc107; 
        margin: 10px 0; 
        color: #000000 !important;
    }

    /* Ensure lists INSIDE suggestions are black */
    .suggestion-box ul, .suggestion-box li, 
    .suggestion-severe ul, .suggestion-severe li {
        color: #000000 !important;
    }

    /* 3. SELECTBOX FIX */
    div[data-baseweb="select"] > div {background-color: #ffffff !important;}
    div[data-baseweb="select"] * {color: #000000 !important;}
    div[data-baseweb="popover"], div[data-baseweb="menu"] {background-color: #ffffff !important;}
    div[data-baseweb="popover"] *, div[data-baseweb="menu"] * {color: #000000 !important;}
    
    /* 4. NEW STYLES FOR V4 */
    .main-header {text-align: center; color: #4F8BF9; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. TRANSLATIONS & MAPPINGS (FROM APP V3 + NEW)
# -----------------------------
translations = {
    "English": {
        "title": "Student Mental Health Assessment",
        "subtitle": "ML-based Screening System",
        "reset_btn": "🔄 Reset System",
        "sidebar_title": "📝 Student Profile (Required)",
        "name": "Student Name (Required)",
        "confirm": "I confirm the profile information is correct",
        "unlock": "✅ Save & Start Assessment",
        "edit_profile": "✏️ Edit Profile",
        "age": "1. Age Group",
        "gender": "2. Gender",
        "uni": "3. University Type",
        "dept": "4. Department",
        "year": "5. Academic Year",
        "cgpa": "6. Current CGPA",
        "scholarship": "7. Scholarship/Waiver?",
        "fill_profile_msg": "🚫 Please complete the student profile on the sidebar to unlock questions.",
        "section_title": "📋 Behavioral Assessment",
        "instructions": "Select one option for each question based on how you felt over the **last 2 weeks**.",
        "radio_opts": ["Not at all", "Sometimes", "Often", "Very Often"],
        "analyze_btn": "🚀 Analyze My Mental Health",
        "analyzing": "Analyzing behavioral patterns...",
        "success": "✅ Assessment Complete",
        "result_title": "📊 Assessment Result",
        "suggestions": "💡 Suggestions",
        "overall_label": "📌 Overall Mental Health Issue:",
        "healthy_msg": "🎉 **Status: Healthy**\nYour responses indicate a balanced mental state. Maintain your current routine.",
        "download_btn": "📥 Download Report",
        "disclaimer_short": "⚠️ This is a screening tool for research purposes, not a clinical diagnosis.",
        "dev_by": "Developed by Team Dual Core",
        "helpline_title": "🆘 Emergency Helpline & Resources",
        "emergency_text": "Your response indicates significant distress. If you feel unsafe, call 999 or a helpline immediately.",
        "clinical_note": "⚠️ **Clinical Note:** Self-harm risk detected despite low overall score.",
        "err_fill": "Please complete all fields correctly.",
        "err_name": "Please enter a valid name (at least 3 letters).",
        # NEW TABS
        "tabs": ["📝 Assessment", "🔍 X-AI Analysis", "🎓 Thesis Artifacts"]
    },
    "Bangla": {
        "title": "শিক্ষার্থী মানসিক স্বাস্থ্য মূল্যায়ন",
        "subtitle": "মেশিন লার্নিং ভিত্তিক স্ক্রিনিং সিস্টেম",
        "reset_btn": "🔄 রিসেট",
        "sidebar_title": "📝 শিক্ষার্থীর প্রোফাইল (আবশ্যক)",
        "name": "শিক্ষার্থীর নাম (আবশ্যক)",
        "confirm": "আমি নিশ্চিত করছি তথ্য সঠিক",
        "unlock": "✅ সেভ করে টেস্ট শুরু করুন",
        "edit_profile": "✏️ প্রোফাইল এডিট করুন",
        "age": "১. বয়স গ্রুপ",
        "gender": "২. লিঙ্গ",
        "uni": "৩. বিশ্ববিদ্যালয়ের ধরণ",
        "dept": "৪. ডিপার্টমেন্ট",
        "year": "৫. শিক্ষাবর্ষ",
        "cgpa": "৬. বর্তমান সিজিপিএ (CGPA)",
        "scholarship": "৭. স্কলারশিপ/ওয়েভার আছে?",
        "fill_profile_msg": "🚫 প্রশ্ন দেখার জন্য দয়া করে বাম পাশের প্রোফাইলটি সম্পূর্ণ পূরণ করুন।",
        "section_title": "📋 আচরণগত মূল্যায়ন",
        "instructions": "গত **২ সপ্তাহের** অনুভূতির ভিত্তিতে প্রতিটি প্রশ্নের জন্য একটি অপশন নির্বাচন করুন।",
        "radio_opts": ["একদম না", "মাঝে মাঝে", "প্রায়ই", "খুব বেশি"],
        "analyze_btn": "🚀 ফলাফল দেখুন",
        "analyzing": "বিশ্লেষণ করা হচ্ছে...",
        "success": "✅ মূল্যায়ন সম্পন্ন",
        "result_title": "📊 ফলাফল",
        "suggestions": "💡 পরামর্শ",
        "overall_label": "📌 সামগ্রিক মানসিক সমস্যা:",
        "healthy_msg": "🎉 **অবস্থা: সুস্থ**\nআপনার মানসিক অবস্থা ভারসাম্যপূর্ণ মনে হচ্ছে। বর্তমান রুটিন বজায় রাখুন।",
        "download_btn": "📥 রিপোর্ট ডাউনলোড",
        "disclaimer_short": "⚠️ এটি একটি স্ক্রিনিং টুল, চিকিৎসার বিকল্প নয়।",
        "dev_by": "ডেভেলপ করেছে Team Dual Core",
        "helpline_title": "🆘 জরুরি হেল্পলাইন ও সেবা",
        "emergency_text": "আপনার উত্তর মানসিক ঝুঁকির ইঙ্গিত দিচ্ছে। নিজেকে আঘাত করার আশঙ্কা থাকলে এখনই ৯৯৯ বা হেল্পলাইনে কল করুন।",
        "clinical_note": "⚠️ **ক্লিনিক্যাল নোট:** সামগ্রিক স্কোর কম হলেও আত্মহানির ঝুঁকি দেখা যাচ্ছে।",
        "err_fill": "সব তথ্য সঠিকভাবে পূরণ করুন।",
        "err_name": "সঠিক নাম লিখুন (অন্তত ৩টি অক্ষর)।",
        # NEW TABS
        "tabs": ["📝 মূল্যায়ন", "🔍 বিশ্লেষণ (X-AI)", "🎓 গবেষণার তথ্য"]
    }
}

# --- INTERNAL OPTIONS & MAPPINGS (Crash Proof) ---
opt_gender = ["Select", "Male", "Female"]
opt_uni = ["Select", "Public", "Private"]
opt_dept = ["Select", "CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"]
opt_year = ["Select", "First Year", "Second Year", "Third Year", "Fourth Year", "Master"]
opt_sch = ["Select", "Yes", "No"]
opt_age = ["Select", "18-22", "23-26", "27-30", "Above 30"]

bn_map = {
    "Select": "সিলেক্ট করুন...", "Male": "পুরুষ", "Female": "মহিলা",
    "Public": "পাবলিক", "Private": "প্রাইভেট",
    "CSE": "সিএসই", "EEE": "ইইই", "BBA": "বিবিএ", "English": "ইংরেজি", "Law": "আইন", "Pharmacy": "ফার্মাসি", "Other": "অন্যান্য",
    "First Year": "১ম বর্ষ", "Second Year": "২য় বর্ষ", "Third Year": "৩য় বর্ষ", "Fourth Year": "৪র্থ বর্ষ", "Master": "মাস্টার্স",
    "Yes": "হ্যাঁ", "No": "না"
}

def format_option(option):
    if st.session_state.get('lang', 'English') == 'Bangla':
        return bn_map.get(option, option)
    return "Select..." if option == "Select" else option

# Questions (26 Items from V3)
q_labels_en = [
    "1. Upset due to academic affairs?", "2. Unable to control important things?", "3. Nervous and stressed?",
    "4. Could not cope with mandatory activities?", "5. Confident about handling problems?", "6. Things going your way?",
    "7. Able to control irritations?", "8. Felt academic performance was on top?", "9. Angered due to bad performance?",
    "10. Difficulties piling up?", "11. Nervous/anxious/on edge?", "12. Unable to stop worrying?",
    "13. Trouble relaxing?", "14. Being so restless?", "15. Easily annoyed/irritable?",
    "16. Afraid something awful might happen?", "17. Worrying too much?", "18. Little interest in doing things?",
    "19. Feeling down/depressed/hopeless?", "20. Trouble sleeping?", "21. Feeling tired/low energy?",
    "22. Poor appetite/overeating?", "23. Feeling bad about yourself?", "24. Trouble concentrating?",
    "25. Moving slowly or too fast?", "26. Thoughts of hurting yourself?"
]
q_labels_bn = [
    "১. পড়াশোনার চাপে মন খারাপ?", "২. নিয়ন্ত্রণে অক্ষম অনুভব?", "৩. নার্ভাস/স্ট্রেস?",
    "৪. বাধ্যতামূলক কাজ সামলাতে কষ্ট?", "৫. সমস্যা সামলাতে আত্মবিশ্বাস?", "৬. সব কিছু আপনার মতো হচ্ছে?",
    "৭. বিরক্তি নিয়ন্ত্রণ করতে পারেন?", "৮. পারফরম্যান্স ভালো মনে হচ্ছে?", "৯. খারাপ ফলাফলে রাগ?",
    "১০. সমস্যা জমে যাচ্ছে মনে হয়?", "১১. উদ্বিগ্ন/অস্থির?", "১২. দুশ্চিন্তা থামাতে পারছেন না?",
    "১৩. রিল্যাক্স করতে সমস্যা?", "১৪. খুব অস্থির লাগে?", "১৫. সহজে বিরক্ত?",
    "১৬. খারাপ কিছু হবে ভয়?", "১৭. বেশি দুশ্চিন্তা?", "১৮. কাজে আগ্রহ কম?",
    "১৯. মন খারাপ/হতাশ?", "২০. ঘুমের সমস্যা?", "২১. ক্লান্ত/শক্তি কম?",
    "২২. ক্ষুধা কম/বেশি খাওয়া?", "২৩. নিজেকে নিয়ে খারাপ লাগে?", "২৪. মনোযোগে সমস্যা?",
    "২৫. খুব ধীর/খুব দ্রুত নড়াচড়া?", "২৬. নিজেকে আঘাত করার চিন্তা?"
]

# -----------------------------
# 3. HELPER FUNCTIONS (V3 + NEW VISUALS)
# -----------------------------
def extract_number(text):
    if not text: return 0.0
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
        return float(match.group()) if match else 0.0
    except: return 0.0

@st.cache_resource
def load_resources():
    try:
        # NOTE: Using Clean Filenames
        model = joblib.load("mental_health_hybrid_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, encoders, feature_columns, None
    except Exception as e:
        return None, None, None, str(e)

def is_low_risk_label(label: str) -> bool:
    low_exact = {"Minimal Anxiety", "Low Stress", "No Depression", "Minimal Depression", "Normal", "None"}
    return (label in low_exact) or any(x in label for x in ["Minimal", "Low", "No Depression", "No Stress", "No Anxiety"])

def severity_bucket(label: str) -> str:
    if any(x in label for x in ["Severe", "High"]): return "Severe/High"
    if "Moderate" in label: return "Moderate"
    return "Mild" 

def get_suggestions(condition: str, bucket: str, lang: str):
    # (Existing Logic from V3 - Shortened for Brevity but functionality remains)
    tips_en = {
        "Anxiety": {
            "Mild": ["Practice controlled breathing exercises (4-7-8).", "Limit caffeine intake.", "Take short breaks outdoors."],
            "Moderate": ["Maintain a worry journal.", "Engage in regular physical activity.", "Reduce screen time before sleep."],
            "Severe/High": ["Talk to a counselor/psychologist today.", "Tell a family member you trust.", "If you feel unsafe, call the helpline immediately."]
        },
        "Stress": {
            "Mild": ["Focus on one task at a time.", "Take short breaks during study.", "Maintain a balanced diet."],
            "Moderate": ["Create a prioritized to-do list.", "Practice muscle relaxation.", "Discuss your academic load with a peer."],
            "Severe/High": ["Seek guidance from an academic advisor.", "Ensure adequate sleep.", "Consider professional stress management."]
        },
        "Depression": {
            "Mild": ["Spend time in natural sunlight.", "Organize your immediate workspace.", "Connect with a friend."],
            "Moderate": ["Engage in a hobby.", "Maintain a regular sleep schedule.", "Set small, achievable daily goals."],
            "Severe/High": ["Seek professional psychological support today.", "Confide in a trusted person.", "Contact emergency services if self-harm thoughts occur."]
        }
    }
    tips_bn = {
        "Anxiety": {
            "Mild": ["নিয়ন্ত্রিত শ্বাস-প্রশ্বাসের ব্যায়াম করুন।", "ক্যাফেইন গ্রহণ সীমিত করুন।", "বাইরে কিছুক্ষণ বিরতি নিন।"],
            "Moderate": ["দুশ্চিন্তাগুলো লিখে রাখুন।", "নিয়মিত শারীরিক ব্যায়াম করুন।", "ঘুমানোর আগে মোবাইল ব্যবহার কমান।"],
            "Severe/High": ["আজই একজন কাউন্সিলর/সাইকোলজিস্টের সাথে কথা বলুন।", "বিশ্বস্ত পরিবারের সদস্যকে জানান।", "নিরাপদ বোধ না করলে এখনই হেল্পলাইনে কল করুন।"]
        },
        "Stress": {
            "Mild": ["একবারে একটি কাজে মনোযোগ দিন।", "পড়ার মাঝে ছোট বিরতি নিন।", "সুষম খাবার গ্রহণ করুন।"],
            "Moderate": ["কাজের অগ্রাধিকার তালিকা তৈরি করুন।", "পেশী শিথিলকরণ ব্যায়াম করুন।", "সহপাঠীর সাথে কথা বলুন।"],
            "Severe/High": ["একাডেমিক অ্যাডভাইজারের পরামর্শ নিন।", "পর্যাপ্ত ঘুম নিশ্চিত করুন।", "পেশাদার সাহায্য নিন।"]
        },
        "Depression": {
            "Mild": ["প্রাকৃতিক রোদে কিছু সময় কাটান।", "নিজের পড়ার টেবিল গুছিয়ে রাখুন।", "বন্ধুর সাথে কথা বলুন।"],
            "Moderate": ["শখের কাজ করুন।", "নিয়মিত ঘুমের রুটিন মেনে চলুন।", "ছোট লক্ষ্য নির্ধারণ করুন।"],
            "Severe/High": ["আজই পেশাদার সাইকোলজিস্টের সাহায্য নিন।", "বিশ্বস্ত কারো সাথে কথা বলুন।", "আত্মহানির চিন্তা এলে এখনই জরুরি সেবায় যোগাযোগ করুন।"]
        }
    }
    dataset = tips_bn if lang == "Bangla" else tips_en
    return dataset.get(condition, {}).get(bucket, dataset.get(condition, {}).get("Mild", []))

# --- NEW VISUALIZATION FUNCTIONS (GAUGE & RADAR) ---
def create_gauge_chart(title, text_label):
    # Convert Text Label to Score
    score = 0
    text_label = str(text_label).lower()
    if "normal" in text_label or "minimal" in text_label or "no " in text_label: score = 0
    elif "mild" in text_label: score = 1
    elif "moderate" in text_label and "severe" not in text_label: score = 2
    elif "moderately severe" in text_label: score = 3
    elif "severe" in text_label or "high" in text_label: score = 4

    colors = ['#66bb6a', '#ffa726', '#ef5350'] # Green, Orange, Red
    color = colors[0] if score <= 1 else (colors[1] if score <= 2 else colors[2])
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
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
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10))
    return fig

def create_radar_chart(answers_list):
    if len(answers_list) < 26: return None
    # Grouping logic
    academic = np.mean(answers_list[0:10])
    anxiety = np.mean(answers_list[10:17])
    depression = np.mean(answers_list[17:26])
    
    categories = ['Academic Pressure', 'Anxiety Symptoms', 'Depression Symptoms', 'General Distress']
    values = [academic, anxiety, depression, np.mean(answers_list)]
    
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Profile'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3])), title="Risk Profile", height=300, margin=dict(t=30, b=20))
    return fig

# -----------------------------
# 4. SESSION MANAGEMENT
# -----------------------------
if "profile_locked" not in st.session_state: st.session_state.profile_locked = False
if "profile_data" not in st.session_state: st.session_state.profile_data = {}
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "latest_answers" not in st.session_state: st.session_state.latest_answers = []

def reset_all():
    st.session_state.clear()
    st.session_state.profile_locked = False
    st.rerun()

# -----------------------------
# 5. UI & LOGIC
# -----------------------------
# Optional Banner
try: st.image("banner.jpg", use_column_width=True)
except: pass

st.sidebar.markdown("### 🌐 Language / ভাষা")
st.session_state.lang = st.sidebar.radio("Language", ("English", "Bangla"), label_visibility="collapsed")
lang = st.session_state.lang
t = translations[lang]

# Title
c1, c2 = st.columns([8, 2])
with c1:
    st.title(t["title"])
    st.caption(t["subtitle"])
with c2:
    if st.button(t["reset_btn"], type="primary"): reset_all()

# Load Model
model, encoders, feature_columns, err = load_resources()
if model is None:
    st.error("🚨 System Error: Model files missing (Check filenames).")
    st.code(err)
    st.stop()

# --- SIDEBAR PROFILE (LOCKED LOGIC FROM V3) ---
st.sidebar.header(t["sidebar_title"])
locked = st.session_state.profile_locked

with st.sidebar.form("profile_form"):
    student_name = st.text_input(t["name"], placeholder="Enter full name", key="p_name", disabled=locked)
    age_input = st.selectbox(t["age"], opt_age, index=0, key="p_age", disabled=locked, format_func=format_option)
    gender_input = st.selectbox(t["gender"], opt_gender, index=0, key="p_gender", disabled=locked, format_func=format_option)
    uni_input = st.selectbox(t["uni"], opt_uni, index=0, key="p_uni", disabled=locked, format_func=format_option)
    dept_input = st.selectbox(t["dept"], opt_dept, index=0, key="p_dept", disabled=locked, format_func=format_option)
    year_input = st.selectbox(t["year"], opt_year, index=0, key="p_year", disabled=locked, format_func=format_option)
    cgpa_input = st.number_input(t["cgpa"], min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f", key="p_cgpa", disabled=locked)
    sch_input = st.selectbox(t["scholarship"], opt_sch, index=0, key="p_sch", disabled=locked, format_func=format_option)
    confirm_ok = st.checkbox(t["confirm"], key="p_conf", disabled=locked)
    lock_btn = st.form_submit_button(t["unlock"], type="primary", disabled=locked)

if locked:
    if st.sidebar.button(t["edit_profile"]):
        st.session_state.profile_locked = False
        st.rerun()

# Validation Logic
name_clean = student_name.strip()
valid_name = len(name_clean) >= 3 and any(c.isalpha() for c in name_clean)
is_valid = lambda x: x != "Select"

if lock_btn:
    if not valid_name:
        st.sidebar.error(t["err_name"])
    elif (is_valid(age_input) and is_valid(gender_input) and is_valid(uni_input) and 
          is_valid(dept_input) and is_valid(year_input) and is_valid(sch_input) and cgpa_input > 0 and confirm_ok):
        st.session_state.profile_data = {
            "name": name_clean, "age": age_input, "gender": gender_input, "uni": uni_input,
            "dept": dept_input, "year": year_input, "cgpa": cgpa_input, "sch": sch_input
        }
        st.session_state.profile_locked = True
        st.rerun()
    else:
        st.sidebar.error(t["err_fill"])

if not st.session_state.profile_locked:
    st.warning(t["fill_profile_msg"])
    st.markdown(f"<div class='locked-hint'>👈 {'Please complete the sidebar profile first.' if lang=='English' else 'দয়া করে বাম পাশের প্রোফাইল পূরণ করুন।'}</div>", unsafe_allow_html=True)
    st.stop()

# --- MAIN APP WITH TABS ---
p_data = st.session_state.profile_data
st.subheader(("👋 Hello, " if lang == "English" else "👋 হ্যালো, ") + p_data["name"])

# TABS DEFINITION
tab1, tab2, tab3 = st.tabs(t['tabs'])

# ========================================================
# TAB 1: ASSESSMENT (LOGIC FROM V3 + VISUALS FROM V4)
# ========================================================
with tab1:
    st.subheader(t["section_title"])
    st.info(t["instructions"])
    
    radio_opts = t["radio_opts"]
    opts_map = {"Not at all": 0, "একদম না": 0, "Sometimes": 1, "মাঝে মাঝে": 1, "Often": 2, "প্রায়ই": 2, "Very Often": 3, "খুব বেশি": 3}
    q_list = q_labels_bn if lang == "Bangla" else q_labels_en
    answers = [0] * len(q_list)
    mid = (len(q_list) + 1) // 2

    cL, cR = st.columns(2)
    with cL:
        for i, q in enumerate(q_list[:mid]):
            val = st.radio(f"**{q}**", radio_opts, horizontal=True, key=f"q_{i}")
            answers[i] = opts_map[val]
            st.divider()
    with cR:
        for i, q in enumerate(q_list[mid:]):
            real_idx = mid + i
            val = st.radio(f"**{q}**", radio_opts, horizontal=True, key=f"q_{real_idx}")
            answers[real_idx] = opts_map[val]
            st.divider()

    analyze = st.button(t["analyze_btn"], type="primary", use_container_width=True)

    # PREDICTION LOGIC
    if analyze:
        input_dict = {
            feature_columns[0]: extract_number(p_data["age"]),
            feature_columns[1]: p_data["gender"],
            feature_columns[2]: p_data["uni"],
            feature_columns[3]: p_data["dept"],
            feature_columns[4]: p_data["year"],
            feature_columns[5]: float(p_data["cgpa"]),
            feature_columns[6]: p_data["sch"]
        }
        for i in range(26):
            input_dict[feature_columns[7+i]] = answers[i]
        
        input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)

        with st.spinner(t["analyzing"]):
            probs = model.predict_proba(input_df)
        
        # SAVE RESULTS TO SESSION STATE FOR TAB 2
        st.session_state.analysis_done = True
        st.session_state.latest_answers = answers
        st.session_state.probs = probs
        st.session_state.answers_raw = answers

    # DISPLAY RESULTS (If Analysis Done)
    if st.session_state.analysis_done:
        probs = st.session_state.probs
        # Check emergency
        if st.session_state.latest_answers[25] >= 2:
            st.markdown(f"<div class='emergency-box'><h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3><p>{t['emergency_text']}</p></div>", unsafe_allow_html=True)
        else:
            st.success(t["success"])

        st.subheader(t["result_title"])

        conds = ["Anxiety", "Stress", "Depression"]
        risk_data = []
        r_txt = ["--- ASSESSMENT REPORT ---", f"Name: {p_data['name']}", f"Date: {datetime.now()}"]

        # VISUAL GAUGE CHARTS ROW
        g_cols = st.columns(3)

        for i, c in enumerate(conds):
            p_arr = probs[i][0]
            idx = int(np.argmax(p_arr))
            lbl = encoders[f"{c} Label"].inverse_transform([idx])[0]
            conf = float(p_arr[idx]) * 100
            
            is_low = is_low_risk_label(lbl)
            bkt = severity_bucket(lbl)
            
            # Display Gauge in Column
            with g_cols[i]:
                st.plotly_chart(create_gauge_chart(c, lbl), use_container_width=True)
                if is_low:
                    st.success(f"**{lbl}** ({conf:.1f}%)")
                    if c == "Depression" and st.session_state.latest_answers[25] >= 2: st.warning(t["clinical_note"])
                else:
                    st.error(f"**{lbl}** ({conf:.1f}%)")

            risk_data.append((c, conf, lbl, bkt, is_low))
            r_txt.append(f"{c}: {lbl} ({conf:.1f}%)")

        # SUGGESTIONS
        st.markdown("---")
        concerns = [r for r in risk_data if not r[4]]
        concerns.sort(key=lambda x: x[1], reverse=True)

        if not concerns:
            st.success(t['healthy_msg'])
        else:
            top_issue = concerns[0]
            st.info(f"**{t['overall_label']} {top_issue[0]} ({top_issue[2]})**", icon="📌")
            st.subheader(t["suggestions"])
            
            for c, conf, lbl, bkt, _ in concerns:
                tips = get_suggestions(c, bkt, lang)
                is_severe = (bkt == "Severe/High")
                style = "suggestion-severe" if is_severe else "suggestion-box"
                
                st.markdown(f"**{c} ({lbl})**")
                st.markdown(f"<div class='{style}'><ul style='margin:0;padding-left:20px'>{''.join([f'<li>{tip}</li>' for tip in tips])}</ul></div>", unsafe_allow_html=True)
                r_txt.extend([f"- {tip}" for tip in tips])

        # EXTRA FEATURES (HELPLINE + BUTTONS)
        st.markdown("---")
        
        # 1. Helpline
        with st.expander(t["helpline_title"]):
            st.markdown("""
            - **Kaan Pete Roi:** 01779554391 (Mental Health Support)
            - **National Emergency:** 999
            - **DIU Counseling Unit:** counseling@diu.edu.bd
            """)

        # 2. Buttons
        b1, b2 = st.columns(2)
        with b1:
            st.download_button(t["download_btn"], data="\n".join(r_txt), file_name=f"Report_{p_data['name']}.txt")
        with b2:
            if st.button("📢 Report Feedback (Dummy)"):
                st.toast("Thank you for your feedback!")

# ========================================================
# TAB 2: X-AI ANALYSIS (RADAR CHART)
# ========================================================
with tab2:
    st.header(t['tabs'][1])
    if st.session_state.analysis_done:
        st.markdown("### Personalized Risk Factor Analysis")
        st.info("This chart visualizes which areas (Academic, Anxiety, etc.) contribute most to your score.")
        
        fig = create_radar_chart(st.session_state.latest_answers)
        if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Please complete the assessment in the first tab to see the analysis.")

# ========================================================
# TAB 3: THESIS ARTIFACTS
# ========================================================
with tab3:
    st.header(t['tabs'][2])
    st.markdown("""
    ### 🔬 Research Methodology
    This system is built upon a **Hybrid Ensemble Classifier** (Logistic Regression + CatBoost).
    
    **Highlights:**
    - **Accuracy:** ~98% (Validated via 5-Fold CV)
    - **Features:** 32 Input parameters (Academic + Behavioral)
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix (Anxiety)**")
        st.caption("*(Upload CM_Anxiety.png in folder to view)*")
        # st.image("CM_Anxiety.png")
    with c2:
        st.markdown("**SHAP Analysis (Feature Importance)**")
        st.caption("*(Upload SHAP_Summary.png in folder to view)*")
        # st.image("SHAP_Summary.png")

# Footer
st.markdown(f"<div class='footer'>{t['dev_by']} | {t['disclaimer_short']}</div>", unsafe_allow_html=True)
