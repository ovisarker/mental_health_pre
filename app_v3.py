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

# Custom CSS
st.markdown("""
<style>
    .footer {text-align:center; padding:20px; font-size:12px; color:#666; border-top:1px solid #ddd; margin-top: 50px;}
    
    .emergency-box {
        background-color: #ffebee; 
        border: 2px solid #ef5350; 
        padding: 15px; 
        border-radius: 10px; 
        color: #c62828 !important; 
        margin: 14px 0;
    }
    
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
    
    .locked-hint {
        background-color: #f8f9fa; 
        border: 1px solid #ddd; 
        padding: 14px; 
        border-radius: 10px; 
        color: #333 !important;
    }
    
    li { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. TRANSLATIONS & MAPPINGS
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
        "helpline_title": "🆘 Emergency Helpline (BD)",
        "emergency_text": "Your response indicates significant distress. If you feel unsafe, call 999 or a helpline immediately.",
        "clinical_note": "⚠️ **Clinical Note:** Self-harm risk detected despite low overall score.",
        "err_fill": "Please complete all fields correctly.",
        "err_name": "Please enter a valid name (at least 3 letters)."
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
        "helpline_title": "🆘 জরুরি হেল্পলাইন (BD)",
        "emergency_text": "আপনার উত্তর মানসিক ঝুঁকির ইঙ্গিত দিচ্ছে। নিজেকে আঘাত করার আশঙ্কা থাকলে এখনই ৯৯৯ বা হেল্পলাইনে কল করুন।",
        "clinical_note": "⚠️ **ক্লিনিক্যাল নোট:** সামগ্রিক স্কোর কম হলেও আত্মহানির ঝুঁকি দেখা যাচ্ছে।",
        "err_fill": "সব তথ্য সঠিকভাবে পূরণ করুন।",
        "err_name": "সঠিক নাম লিখুন (অন্তত ৩টি অক্ষর)।"
    }
}

# --- INTERNAL OPTIONS & MAPPINGS (The Fix) ---
# Internal values are English (Canonical). Mappings are for Display.
opt_gender = ["Select", "Male", "Female"]
opt_uni = ["Select", "Public", "Private"]
opt_dept = ["Select", "CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"]
opt_year = ["Select", "First Year", "Second Year", "Third Year", "Fourth Year", "Master"]
opt_sch = ["Select", "Yes", "No"]
opt_age = ["Select", "18-22", "23-26", "27-30", "Above 30"]

# Display Mappings
bn_map = {
    "Select": "সিলেক্ট করুন...",
    "Male": "পুরুষ", "Female": "মহিলা",
    "Public": "পাবলিক", "Private": "প্রাইভেট",
    "CSE": "সিএসই", "EEE": "ইইই", "BBA": "বিবিএ", "English": "ইংরেজি", "Law": "আইন", "Pharmacy": "ফার্মাসি", "Other": "অন্যান্য",
    "First Year": "১ম বর্ষ", "Second Year": "২য় বর্ষ", "Third Year": "৩য় বর্ষ", "Fourth Year": "৪র্থ বর্ষ", "Master": "মাস্টার্স",
    "Yes": "হ্যাঁ", "No": "না"
}

def format_option(option):
    # Returns Bangla label if language is Bangla, else returns the option itself
    if st.session_state.get('lang', 'English') == 'Bangla':
        return bn_map.get(option, option)
    return "Select..." if option == "Select" else option

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
    return "Mild" 

def get_suggestions(condition: str, bucket: str, lang: str):
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
        },
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
        },
    }
    dataset = tips_bn if lang == "Bangla" else tips_en
    return dataset.get(condition, {}).get(bucket, dataset.get(condition, {}).get("Mild", []))

# -----------------------------
# 4. SESSION MANAGEMENT
# -----------------------------
if "profile_locked" not in st.session_state:
    st.session_state.profile_locked = False
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}

def reset_all():
    st.session_state.clear()
    st.session_state.profile_locked = False
    st.rerun()

# -----------------------------
# 5. UI & LOGIC
# -----------------------------
st.sidebar.markdown("### 🌐 Language / ভাষা")
# Store lang in session state so format_func can access it
st.session_state.lang = st.sidebar.radio("Language", ("English", "Bangla"), label_visibility="collapsed")
lang = st.session_state.lang
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

locked = st.session_state.profile_locked

with st.sidebar.form("profile_form"):
    # Using format_func for Bilingual Options (Crash-Proof)
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

# Edit Button Logic
if locked:
    if st.sidebar.button(t["edit_profile"]):
        st.session_state.profile_locked = False
        st.rerun()

# Validation logic
name_clean = student_name.strip()
valid_name = len(name_clean) >= 3 and any(c.isalpha() for c in name_clean)
# Check against "Select" (internal value)
is_valid = lambda x: x != "Select"

if lock_btn:
    if not valid_name:
        st.sidebar.error(t["err_name"])
    elif (is_valid(age_input) and is_valid(gender_input) and 
          is_valid(uni_input) and is_valid(dept_input) and is_valid(year_input) and 
          is_valid(sch_input) and cgpa_input > 0 and confirm_ok):
        
        # Save validated data to session state
        st.session_state.profile_data = {
            "name": name_clean,
            "age": age_input,
            "gender": gender_input,
            "uni": uni_input,
            "dept": dept_input,
            "year": year_input,
            "cgpa": cgpa_input,
            "sch": sch_input
        }
        st.session_state.profile_locked = True
        st.rerun()
    else:
        st.sidebar.error(t["err_fill"])

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
# Use Saved Data for Display (Greeting)
p_data = st.session_state.profile_data

st.subheader(("👋 Hello, " if lang == "English" else "👋 হ্যালো, ") + p_data["name"])
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

# --- DIRECT RENDERING ---
cL, cR = st.columns(2)
for i, q in enumerate(q_list):
    with (cL if i % 2 == 0 else cR):
        # Stable key: survives language switch
        val = st.radio(f"**{q}**", radio_opts, horizontal=True, key=f"q_{i}")
        answers.append(opts_map[val])
        st.divider()

analyze = st.button(t["analyze_btn"], type="primary", use_container_width=True)

# --- RESULTS ---
if analyze:
    # Use p_data (Internal English Values) directly for prediction
    input_dict = {
        feature_columns[0]: extract_number(p_data["age"]),
        feature_columns[1]: p_data["gender"], # Already "Male"/"Female"
        feature_columns[2]: p_data["uni"],    # Already "Public"/"Private"
        feature_columns[3]: p_data["dept"],   # Already "CSE", etc.
        feature_columns[4]: p_data["year"],   # Already "First Year", etc.
        feature_columns[5]: float(p_data["cgpa"]),
        feature_columns[6]: p_data["sch"]     # Already "Yes"/"No"
    }
    for i in range(26):
        input_dict[feature_columns[7+i]] = answers[i]
    
    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)

    with st.spinner(t["analyzing"]):
        probs = model.predict_proba(input_df)

    if answers[25] >= 2:
        st.markdown(f"<div class='emergency-box'><h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3><p>{t['emergency_text']}</p></div>", unsafe_allow_html=True)

    st.success(t["success"])
    st.subheader(t["result_title"])

    conds = ["Anxiety", "Stress", "Depression"]
    cards = st.columns(3)
    risk_data = [] 
    
    r_txt = [
        "--- ASSESSMENT REPORT ---",
        f"Name: {p_data['name']}",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Profile: {p_data['gender']}, {p_data['dept']}, CGPA {p_data['cgpa']:.2f}",
        "-----------------------"
    ]

    for i, c in enumerate(conds):
        p_arr = probs[i][0]
        idx = int(np.argmax(p_arr))
        lbl = encoders[f"{c} Label"].inverse_transform([idx])[0]
        conf = float(p_arr[idx]) * 100
        
        is_low = is_low_risk_label(lbl)
        bkt = severity_bucket(lbl)

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

    # --- SUGGESTIONS ---
    st.markdown("---")
    
    concerns = [r for r in risk_data if not r[4]] 
    concerns.sort(key=lambda x: x[1], reverse=True) 

    if not concerns:
        st.success(t['healthy_msg'])
        r_txt.append("\nOverall: Healthy/Balanced state.")
    else:
        top_issue = concerns[0] 
        overall_text = f"**{t['overall_label']} {top_issue[0]} ({top_issue[2]})**"
        st.info(overall_text, icon="📌")
        r_txt.append(f"\n{t['overall_label']} {top_issue[0]} ({top_issue[2]})")

        st.subheader(t["suggestions"])
        
        for c, conf, lbl, bkt, _ in concerns:
            tips = get_suggestions(c, bkt, lang)
            is_severe = (bkt == "Severe/High") or (c == "Depression" and answers[25] >= 2)
            style = "suggestion-severe" if is_severe else "suggestion-box"
            
            st.markdown(f"**{c} ({lbl})**")
            st.markdown(f"<div class='{style}' style='color:black;'><ul style='margin:0;padding-left:20px'>{''.join([f'<li>{tip}</li>' for tip in tips])}</ul></div>", unsafe_allow_html=True)
            
            r_txt.append(f"\n[{c} Suggestions]")
            r_txt.extend([f"- {tip}" for tip in tips])

    st.markdown("---")
    st.download_button(
        label=t["download_btn"],
        data="\n".join(r_txt),
        file_name=f"Report_{p_data['name'].replace(' ', '_')}.txt",
        mime="text/plain"
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown(
    f"<div class='footer'>{t['dev_by']} | {t['disclaimer_short']}</div>",
    unsafe_allow_html=True
)
