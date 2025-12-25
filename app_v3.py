import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Student Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished UI and hiding unnecessary labels
st.markdown("""
<style>
    .footer {text-align:center; padding:20px; font-size:12px; color:#666; border-top:1px solid #ddd; margin-top: 50px;}
    .emergency-box {background:#ffebee; border:2px solid #ef5350; padding:15px; border-radius:10px; color:#c62828; margin:14px 0;}
    .suggestion-friendly {background:#e3f2fd; padding:14px; border-radius:10px; border-left:5px solid #2196f3; margin:10px 0; color: #0d47a1;}
    .suggestion-serious {background:#fff3e0; padding:14px; border-radius:10px; border-left:5px solid #ff9800; margin:10px 0; color: #e65100;}
    .locked-hint {background:#f8f9fa; border:1px solid #ddd; padding:14px; border-radius:10px; color: #555;}
    /* Hide radio button labels if they are empty to prevent warnings */
    div[data-testid="stRadio"] > label {display: none;} 
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. TRANSLATIONS & TEXT ASSETS
# -----------------------------
translations = {
    "English": {
        "title": "Student Mental Health Assessment",
        "subtitle": "ML-based Screening System (Bilingual)",
        "lang_label": "🌐 Language / ভাষা",
        "reset_btn": "🔄 Reset System",
        "sidebar_title": "📝 Student Profile (Required)",
        "name": "Student Name (Required)",
        "confirm": "I confirm the profile information is correct",
        "unlock": "✅ Save & Start Assessment",
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
        "result_title": "📊 Your Wellness Result",
        "suggestions": "💡 Friendly Suggestions",
        "healthy_msg": "🎉 **You are doing great!**\nYour responses indicate a balanced mental state. Keep maintaining your routine and connection with others.",
        "download_btn": "📥 Download Report",
        "disclaimer_short": "⚠️ This is a screening tool for research purposes, not a clinical diagnosis.",
        "dev_by": "Developed by Team Dual Core",
        "helpline_title": "🆘 Emergency Helpline (BD)",
        "emergency_text": "Your response indicates significant distress. Please contact a counselor/psychologist or call the helpline immediately.",
        "clinical_note": "⚠️ **Clinical Note:** Self-harm risk detected despite low overall score.",
        "select": "Select...",
        "genders": ["Select...", "Male", "Female"],
        "unis": ["Select...", "Public", "Private"],
        "scholars": ["Select...", "Yes", "No"],
        "years": ["Select...", "First Year", "Second Year", "Third Year", "Fourth Year", "Master"],
        "depts": ["Select...", "CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"],
        "ages": ["Select...", "18-22", "23-26", "27-30", "Above 30"],
    },
    "Bangla": {
        "title": "শিক্ষার্থী মানসিক স্বাস্থ্য মূল্যায়ন",
        "subtitle": "মেশিন লার্নিং ভিত্তিক স্ক্রিনিং (দ্বিভাষিক)",
        "lang_label": "🌐 Language / ভাষা",
        "reset_btn": "🔄 রিসেট",
        "sidebar_title": "📝 শিক্ষার্থীর প্রোফাইল (আবশ্যক)",
        "name": "শিক্ষার্থীর নাম (আবশ্যক)",
        "confirm": "আমি নিশ্চিত করছি তথ্য সঠিক",
        "unlock": "✅ সেভ করে টেস্ট শুরু করুন",
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
        "result_title": "📊 আপনার ফলাফল",
        "suggestions": "💡 পরামর্শ",
        "healthy_msg": "🎉 **আপনি দারুণ আছেন!**\nআপনার মানসিক অবস্থা ভারসাম্যপূর্ণ মনে হচ্ছে। নিজের যত্ন নেওয়া চালিয়ে যান।",
        "download_btn": "📥 রিপোর্ট ডাউনলোড",
        "disclaimer_short": "⚠️ এটি একটি স্ক্রিনিং টুল, চিকিৎসার বিকল্প নয়।",
        "dev_by": "ডেভেলপ করেছে Team Dual Core",
        "helpline_title": "🆘 জরুরি হেল্পলাইন (BD)",
        "emergency_text": "আপনার উত্তর মানসিক ঝুঁকির ইঙ্গিত দিচ্ছে। দয়া করে কাউন্সিলর/সাইকোলজিস্টের সাথে কথা বলুন অথবা হেল্পলাইনে কল করুন।",
        "clinical_note": "⚠️ **ক্লিনিক্যাল নোট:** সামগ্রিক স্কোর কম হলেও আত্মহানির ঝুঁকি দেখা যাচ্ছে।",
        "select": "সিলেক্ট করুন...",
        "genders": ["সিলেক্ট করুন...", "পুরুষ", "মহিলা"],
        "unis": ["সিলেক্ট করুন...", "পাবলিক", "প্রাইভেট"],
        "scholars": ["সিলেক্ট করুন...", "হ্যাঁ", "না"],
        "years": ["সিলেক্ট করুন...", "১ম বর্ষ", "২য় বর্ষ", "৩য় বর্ষ", "৪র্থ বর্ষ", "মাস্টার্স"],
        "depts": ["সিলেক্ট করুন...", "সিএসই", "ইইই", "বিবিএ", "ইংরেজি", "আইন", "ফার্মাসি", "অন্যান্য"],
        "ages": ["সিলেক্ট করুন...", "18-22", "23-26", "27-30", "Above 30"],
    }
}

# Questions
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
# 3. HELPER FUNCTIONS
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
    return "Mild" # Default fallback for low/minimal

def friendly_tips(condition: str, bucket: str, lang: str):
    tips_en = {
        "Anxiety": {
            "Mild": ["👋 **Hey!** Take 5 slow breaths. Drink some water.", "☕ **Chill:** Skip the extra coffee today.", "📞 **Connect:** Talk to a friend for 5 minutes."],
            "Moderate": ["📓 **Journal:** Write down your worries and close the notebook.", "🏃 **Move:** Go for a 10-15 min walk.", "📵 **Unplug:** Avoid phone screens 30 mins before sleep."],
            "Severe/High": ["🩺 **Counselor:** Please talk to a university counselor.", "🆘 **Support:** Don't be alone right now—reach out to family.", "🚨 **Safety:** If you feel unsafe, call the helpline immediately."]
        },
        "Stress": {
            "Mild": ["🎬 **Break:** Watch something funny or listen to music.", "📅 **Focus:** Do just one small task, don't overthink.", "🍕 **Treat:** Eat a healthy meal you enjoy."],
            "Moderate": ["📝 **Plan:** Make a simple to-do list (Top 3 tasks only).", "🧘 **Relax:** Try light stretching or yoga.", "🗣️ **Share:** Ask for help instead of carrying it all alone."],
            "Severe/High": ["🛑 **Pause:** You might be burning out. Take a break urgently.", "🩺 **Advisor:** Talk to your academic advisor or counselor.", "🚑 **Health:** High stress affects health. Prioritize rest."]
        },
        "Depression": {
            "Mild": ["🌞 **Sunlight:** Open the curtains or go outside for 10 mins.", "🧹 **Tidy:** Clean one small area (like your desk).", "💬 **Message:** Text a friend you trust."],
            "Moderate": ["🚶 **Walk:** A short walk can help lift your mood.", "🎨 **Hobby:** Do something small you used to enjoy.", "📅 **Routine:** Stick to a simple routine for today."],
            "Severe/High": ["🩺 **Professional:** Please see a psychologist or counselor.", "👨‍👩‍👧 **Family:** Tell a family member how you are feeling.", "🆘 **Urgent:** If you have thoughts of self-harm, call the helpline immediately."]
        },
    }
    tips_bn = {
        "Anxiety": {
            "Mild": ["👋 **হেই!** ৫ বার ধীরে শ্বাস নিন ও পানি পান করুন।", "☕ **চিল:** আজ ক্যাফেইন কমান।", "📞 **কথা বলুন:** বন্ধুর সাথে ৫ মিনিট কথা বলুন।"],
            "Moderate": ["📓 **লিখুন:** দুশ্চিন্তাগুলো লিখে খাতা বন্ধ করে রাখুন।", "🏃 **হাঁটুন:** ১০–১৫ মিনিট হাঁটুন।", "📵 **ফোন দূরে:** ঘুমানোর আগে ৩০ মিনিট ফোন ব্যবহার করবেন না।"],
            "Severe/High": ["🩺 **কাউন্সিলর:** দয়া করে কাউন্সিলর/সাইকোলজিস্টের সাথে কথা বলুন।", "🆘 **সাহায্য:** একাই থাকবেন না—পরিবার/বন্ধুকে জানান।", "🚨 **নিরাপত্তা:** ঝুঁকি মনে হলে সাথে সাথে হেল্পলাইনে কল করুন।"]
        },
        "Stress": {
            "Mild": ["🎬 **ব্রেক:** মজার কিছু দেখুন বা গান শুনুন।", "📅 **ফোকাস:** একবারে সব না—আজ শুধু ১টা ছোট কাজ করুন।", "🍕 **খাবার:** পছন্দের স্বাস্থ্যকর খাবার খান।"],
            "Moderate": ["📝 **প্ল্যান:** Top-3 টু-ডু লিস্ট করুন।", "🧘 **রিলাক্স:** হালকা স্ট্রেচিং বা যোগব্যায়াম করুন।", "🗣️ **শেয়ার:** কারো সাহায্য নিন—একাই সব চাপ নেবেন না।"],
            "Severe/High": ["🛑 **থামুন:** অতিরিক্ত চাপে ক্ষতি হচ্ছে—ব্রেক নিন।", "🩺 **অ্যাডভাইজার:** অ্যাডভাইজার বা কাউন্সিলরকে জানান।", "🚑 **স্বাস্থ্য:** বিশ্রাম জরুরি। খুব খারাপ লাগলে ডাক্তার দেখান।"]
        },
        "Depression": {
            "Mild": ["🌞 **রোদ:** পর্দা খুলে ১০ মিনিট রোদ নিন।", "🧹 **গোছানো:** টেবিল বা বিছানা একটু গুছান।", "💬 **মেসেজ:** বিশ্বাসভাজন কাউকে মেসেজ দিন।"],
            "Moderate": ["🚶 **হাঁটা:** বাইরে একটু হেঁটে আসুন।", "🎨 **শখ:** ছোট একটা শখের কাজ করুন (গান/গেম)।", "📅 **রুটিন:** আজকের জন্য ছোট রুটিন ঠিক করুন।"],
            "Severe/High": ["🩺 **পেশাদার:** দয়া করে সাইকোলজিস্ট দেখান।", "👨‍👩‍👧 **পরিবার:** পরিবারের কাউকে এখনই জানান।", "🆘 **জরুরি:** আত্মহানির চিন্তা থাকলে সাথে সাথে হেল্পলাইনে কল করুন।"]
        },
    }
    dataset = tips_bn if lang == "Bangla" else tips_en
    return dataset.get(condition, {}).get(bucket, dataset.get(condition, {}).get("Mild", []))

# -----------------------------
# 4. SESSION MANAGEMENT
# -----------------------------
if "profile_locked" not in st.session_state:
    st.session_state.profile_locked = False

def reset_all():
    st.session_state.clear()
    st.session_state.profile_locked = False
    st.rerun()

# -----------------------------
# 5. UI & LOGIC
# -----------------------------
st.sidebar.markdown("### 🌐 Language / ভাষা")
lang = st.sidebar.radio("Language", ("English", "Bangla"), label_visibility="collapsed")
t = translations[lang]

# Title
c1, c2 = st.columns([8, 2])
with c1:
    st.title(t["title"])
    st.caption(t["subtitle"])
with c2:
    if st.button(t["reset_btn"], type="primary"):
        reset_all()
st.markdown("---")

# Load Model
model, encoders, feature_columns, err = load_resources()
if model is None:
    st.error("🚨 System Error: Model files missing.")
    st.code(err)
    st.stop()

# --- SIDEBAR PROFILE ---
st.sidebar.header(t["sidebar_title"])

with st.sidebar.form("profile_form"):
    student_name = st.text_input(t["name"], placeholder=("Enter full name" if lang == "English" else "পূর্ণ নাম লিখুন"))
    
    age_input = st.selectbox(t["age"], t["ages"], index=0)
    gender_input = st.selectbox(t["gender"], t["genders"], index=0)
    uni_input = st.selectbox(t["uni"], t["unis"], index=0)
    dept_input = st.selectbox(t["dept"], t["depts"], index=0)
    year_input = st.selectbox(t["year"], t["years"], index=0)
    cgpa_input = st.number_input(t["cgpa"], min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f")
    sch_input = st.selectbox(t["scholarship"], t["scholars"], index=0)

    confirm_ok = st.checkbox(t["confirm"])
    lock_btn = st.form_submit_button(t["unlock"], type="primary")

# Validation logic
sentinels = {t["select"], "Select...", "সিলেক্ট করুন..."}
def is_valid(x): return x and (x not in sentinels) and (not str(x).startswith("Select"))

if lock_btn:
    # Check strict validation
    if (student_name.strip() and 
        is_valid(age_input) and 
        is_valid(gender_input) and 
        is_valid(uni_input) and 
        is_valid(dept_input) and 
        is_valid(year_input) and 
        is_valid(sch_input) and 
        cgpa_input > 0 and 
        confirm_ok):
        
        st.session_state.profile_locked = True
        st.sidebar.success("✅ Profile Saved!" if lang == "English" else "✅ প্রোফাইল সেভ হয়েছে!")
    else:
        st.session_state.profile_locked = False
        st.sidebar.error("Please complete all fields correctly." if lang == "English" else "সব তথ্য সঠিকভাবে পূরণ করুন।")

# Helpline
with st.sidebar.expander(t["helpline_title"], expanded=True):
    st.markdown("""
📞 **Kaan Pete Roi:** 01779554391  
📞 **Moner Bondhu:** 01779632588  
🚑 **National Emergency:** 999
""")

# Gatekeeper
if not st.session_state.profile_locked:
    st.warning(t["fill_profile_msg"])
    st.markdown(f"<div class='locked-hint'>👈 {'Please complete the sidebar profile first.' if lang=='English' else 'দয়া করে বাম পাশের প্রোফাইল পূরণ করুন।'}</div>", unsafe_allow_html=True)
    st.stop()

# --- QUESTIONNAIRE ---
# Mapping inputs for model
gender_model = "Male" if gender_input in ["Male", "পুরুষ"] else "Female"
uni_model = "Public" if uni_input in ["Public", "পাবলিক"] else "Private"
sch_model = "Yes" if sch_input in ["Yes", "হ্যাঁ"] else "No"
dept_map = {"সিএসই": "CSE", "ইইই": "EEE", "বিবিএ": "BBA", "ইংরেজি": "English", "আইন": "Law", "ফার্মাসি": "Pharmacy", "অন্যান্য": "Other"}
dept_model = dept_map.get(dept_input, dept_input)
year_map = {"১ম বর্ষ": "First Year", "২য় বর্ষ": "Second Year", "৩য় বর্ষ": "Third Year", "৪র্থ বর্ষ": "Fourth Year", "মাস্টার্স": "Master"}
year_model = year_map.get(year_input, year_input)

st.subheader(("👋 Hello, " if lang == "English" else "👋 হ্যালো, ") + student_name.strip())
st.subheader(t["section_title"])
st.info(t["instructions"])

radio_opts = t["radio_opts"]
opts_map = {
    "Not at all": 0, "একদম না": 0,
    "Sometimes": 1, "মাঝে মাঝে": 1,
    "Often": 2, "প্রায়ই": 2,
    "Very Often": 3, "খুব বেশি": 3
}
q_list = q_labels_bn if lang == "Bangla" else q_labels_en
answers = []

with st.form("qs_form"):
    cL, cR = st.columns(2)
    for i, q in enumerate(q_list):
        with (cL if i % 2 == 0 else cR):
            # Using unique key per question
            val = st.radio(f"**{q}**", radio_opts, horizontal=True, key=f"q_{i}")
            answers.append(opts_map[val])
            st.divider()
    analyze = st.form_submit_button(t["analyze_btn"], type="primary", use_container_width=True)

# --- RESULTS ---
if analyze:
    # Prepare input dataframe
    input_dict = {
        feature_columns[0]: extract_number(age_input),
        feature_columns[1]: gender_model,
        feature_columns[2]: uni_model,
        feature_columns[3]: dept_model,
        feature_columns[4]: year_model,
        feature_columns[5]: float(cgpa_input),
        feature_columns[6]: sch_model
    }
    for i in range(26):
        input_dict[feature_columns[7+i]] = answers[i]
    
    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)

    with st.spinner(t["analyzing"]):
        probs = model.predict_proba(input_df)

    # Q26 Safety check
    if answers[25] >= 2:
        st.markdown(f"<div class='emergency-box'><h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3><p>{t['emergency_text']}</p></div>", unsafe_allow_html=True)

    st.success(t["success"])
    st.subheader(t["result_title"])

    conds = ["Anxiety", "Stress", "Depression"]
    cards = st.columns(3)
    risk_data = [] # (cond, conf, label, bucket, is_low)
    
    # Report text builder
    r_txt = [
        "--- WELLNESS REPORT ---",
        f"Name: {student_name}",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Profile: {gender_model}, {dept_model}, CGPA {cgpa_input:.2f}",
        "-----------------------"
    ]

    for i, c in enumerate(conds):
        p_arr = probs[i][0]
        idx = int(np.argmax(p_arr))
        lbl = encoders[f"{c} Label"].inverse_transform([idx])[0]
        conf = float(p_arr[idx]) * 100
        
        is_low = is_low_risk_label(lbl)
        bkt = severity_bucket(lbl)

        # Translated label display
        d_lbl = lbl
        if lang == "Bangla":
            if is_low: d_lbl = "ঝুঁকি নেই / কম"
            elif "Moderate" in lbl: d_lbl = "মাঝারি"
            elif any(x in lbl for x in ["Severe", "High"]): d_lbl = "তীব্র"
            else: d_lbl = "মৃদু"
        else:
            if is_low: d_lbl = f"No/Low {c}"

        with cards[i]:
            st.markdown(f"### {c}")
            if is_low:
                st.success(f"**{d_lbl}**")
                st.progress(0)
                if c == "Depression" and answers[25] >= 2:
                    st.warning(t["clinical_note"])
            else:
                st.error(f"**{d_lbl}**")
                st.progress(min(100, max(1, int(conf))))
            st.caption(f"Confidence: {conf:.1f}%")
        
        r_txt.append(f"{c}: {lbl} ({conf:.1f}%)")
        risk_data.append((c, conf, lbl, bkt, is_low))

    # Suggestions
    st.markdown("---")
    st.subheader(t["suggestions"])
    
    # Filter for concerns
    concerns = [r for r in risk_data if not r[4]]
    concerns.sort(key=lambda x: x[1], reverse=True)

    if not concerns:
        st.markdown(f"<div class='suggestion-friendly'>{t['healthy_msg']}</div>", unsafe_allow_html=True)
        r_txt.append("\nOverall: Healthy/Balanced state.")
    else:
        for c, conf, lbl, bkt, _ in concerns:
            tips = friendly_tips(c, bkt, lang)
            is_serious = (bkt == "Severe/High") or (c == "Depression" and answers[25] >= 2)
            style = "suggestion-serious" if is_serious else "suggestion-friendly"
            
            st.markdown(f"##### 👉 **{c}**")
            st.markdown(f"<div class='{style}'><ul style='margin:0;padding-left:20px'>{''.join([f'<li>{tip}</li>' for tip in tips])}</ul></div>", unsafe_allow_html=True)
            
            r_txt.append(f"\n[{c} Tips]")
            r_txt.extend([f"- {re.sub(r'[*]', '', tip)}" for tip in tips])

    st.markdown("---")
    st.download_button(
        label=t["download_btn"],
        data="\n".join(r_txt),
        file_name=f"Report_{student_name.replace(' ', '_')}.txt",
        mime="text/plain"
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"<div class='footer'>{t['dev_by']} | {t['disclaimer_short']}</div>", unsafe_allow_html=True)
