import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Student Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .footer {text-align:center; padding:20px; font-size:12px; color:#666; border-top:1px solid #222;}
    .emergency-box {background:#ffebee; border:2px solid #ef5350; padding:15px; border-radius:10px; color:#c62828; margin:14px 0;}
    .suggestion-friendly {background:#e3f2fd; padding:14px; border-radius:10px; border-left:5px solid #2196f3; margin:10px 0;}
    .suggestion-serious {background:#fff3e0; padding:14px; border-radius:10px; border-left:5px solid #ff9800; margin:10px 0;}
    .locked-hint {background:#111; border:1px solid #333; padding:14px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TRANSLATIONS
# -----------------------------
translations = {
    "English": {
        "title": "Student Mental Health Assessment",
        "subtitle": "ML-based Screening (Bilingual)",
        "lang_label": "🌐 Language / ভাষা",
        "reset_btn": "🔄 Reset",
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
        "fill_profile_msg": "🚫 Please complete the profile (and confirm) to unlock questions.",
        "section_title": "📋 Behavioral Assessment",
        "instructions": "Select one option for each question based on the last 2 weeks.",
        "radio_opts": ["Not at all", "Sometimes", "Often", "Very Often"],
        "analyze_btn": "🚀 Analyze My Mental Health",
        "analyzing": "Analyzing patterns...",
        "success": "✅ Assessment Complete",
        "result_title": "📊 Your Wellness Result",
        "suggestions": "💡 Friendly Suggestions",
        "healthy_msg": "🎉 You look okay overall. Keep maintaining your routine and balance.",
        "download_btn": "📥 Download Report",
        "disclaimer_short": "⚠️ This is a screening tool, not a medical diagnosis.",
        "dev_by": "Developed by Team Dual Core",
        "helpline_title": "🆘 Emergency Helpline (BD)",
        "emergency_text": "Your response indicates distress. Please contact a counselor/psychologist or call helpline immediately.",
        "clinical_note": "⚠️ Clinical Note: Self-harm risk detected despite low overall score.",
        "select": "Select...",
        "genders": ["Select...", "Male", "Female"],
        "unis": ["Select...", "Public", "Private"],
        "scholars": ["Select...", "Yes", "No"],
        "years": ["Select...", "First Year", "Second Year", "Third Year", "Fourth Year", "Master"],
        "depts": ["Select...", "CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"],
        "ages": ["Select...", "18-22", "23-26", "27-30", "Above 30"],
    },
    "Bangla": {
        "title": "শিক্ষার্থী মানসিক স্বাস্থ্য মূল্যায়ন",
        "subtitle": "এমএল ভিত্তিক স্ক্রিনিং (দ্বিভাষিক)",
        "lang_label": "🌐 Language / ভাষা",
        "reset_btn": "🔄 রিসেট",
        "sidebar_title": "📝 শিক্ষার্থীর প্রোফাইল (অবশ্যই পূরণীয়)",
        "name": "শিক্ষার্থীর নাম (অবশ্যই)",
        "confirm": "আমি নিশ্চিত করছি প্রোফাইল তথ্য সঠিক",
        "unlock": "✅ সেভ করে টেস্ট শুরু করুন",
        "age": "১. বয়স গ্রুপ",
        "gender": "২. লিঙ্গ",
        "uni": "৩. বিশ্ববিদ্যালয়ের ধরণ",
        "dept": "৪. ডিপার্টমেন্ট",
        "year": "৫. শিক্ষাবর্ষ",
        "cgpa": "৬. বর্তমান সিজিপিএ (CGPA)",
        "scholarship": "৭. স্কলারশিপ/ওয়েভার আছে?",
        "fill_profile_msg": "🚫 প্রশ্ন দেখতে হলে প্রোফাইল সম্পূর্ণ পূরণ করে কনফার্ম করুন।",
        "section_title": "📋 আচরণগত মূল্যায়ন",
        "instructions": "গত ২ সপ্তাহের ভিত্তিতে প্রতিটি প্রশ্নে একটি অপশন নির্বাচন করুন।",
        "radio_opts": ["একদম না", "মাঝে মাঝে", "প্রায়ই", "খুব বেশি"],
        "analyze_btn": "🚀 ফলাফল দেখুন",
        "analyzing": "বিশ্লেষণ চলছে...",
        "success": "✅ মূল্যায়ন সম্পন্ন",
        "result_title": "📊 আপনার ফলাফল",
        "suggestions": "💡 বন্ধুসুলভ পরামর্শ",
        "healthy_msg": "🎉 মোটের উপর ঠিক আছে। রুটিন ও ব্যালান্স বজায় রাখুন।",
        "download_btn": "📥 রিপোর্ট ডাউনলোড",
        "disclaimer_short": "⚠️ এটি একটি স্ক্রিনিং টুল, চিকিৎসার বিকল্প নয়।",
        "dev_by": "ডেভেলপ করেছে Team Dual Core",
        "helpline_title": "🆘 জরুরি হেল্পলাইন (BD)",
        "emergency_text": "আপনার উত্তর মানসিক ঝুঁকির ইঙ্গিত দিচ্ছে। দয়া করে কাউন্সিলর/সাইকোলজিস্টের সাথে কথা বলুন অথবা হেল্পলাইনে কল করুন।",
        "clinical_note": "⚠️ ক্লিনিক্যাল নোট: সামগ্রিক স্কোর কম হলেও আত্মহানির ঝুঁকি দেখা যাচ্ছে।",
        "select": "সিলেক্ট করুন...",
        "genders": ["সিলেক্ট করুন...", "পুরুষ", "মহিলা"],
        "unis": ["সিলেক্ট করুন...", "পাবলিক", "প্রাইভেট"],
        "scholars": ["সিলেক্ট করুন...", "হ্যাঁ", "না"],
        "years": ["সিলেক্ট করুন...", "১ম বর্ষ", "২য় বর্ষ", "৩য় বর্ষ", "৪র্থ বর্ষ", "মাস্টার্স"],
        "depts": ["সিলেক্ট করুন...", "সিএসই", "ইইই", "বিবিএ", "ইংরেজি", "আইন", "ফার্মাসি", "অন্যান্য"],
        "ages": ["সিলেক্ট করুন...", "18-22", "23-26", "27-30", "Above 30"],
    }
}

# Short question labels (keep yours if you want)
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
    "১. পড়াশোনার চাপে মন খারাপ?", "২. নিয়ন্ত্রণে অক্ষম অনুভব?", "৩. নার্ভাস/স্ট্রেস?",
    "৪. বাধ্যতামূলক কাজ সামলাতে কষ্ট?", "৫. সমস্যা সামলাতে আত্মবিশ্বাস?", "৬. সব কিছু আপনার মতো হচ্ছে?",
    "৭. বিরক্তি নিয়ন্ত্রণ করতে পারেন?", "৮. পারফরম্যান্স ভালো মনে হচ্ছে?", "৯. খারাপ ফলাফলে রাগ?",
    "১০. সমস্যা জমে যাচ্ছে মনে হয়?", "১১. উদ্বিগ্ন/অস্থির?", "১২. দুশ্চিন্তা থামাতে পারছেন না?",
    "১৩. রিল্যাক্স করতে সমস্যা?", "১৪. খুব অস্থির লাগে?", "১৫. সহজে বিরক্ত?",
    "১৬. খারাপ কিছু হবে ভয়?", "১৭. বেশি দুশ্চিন্তা?", "১৮. কাজে আগ্রহ কম?",
    "১৯. মন খারাপ/হতাশ?", "২০. ঘুমের সমস্যা?", "২১. ক্লান্ত/শক্তি কম?",
    "২২. ক্ষুধা কম/বেশি খাওয়া?", "২৩. নিজেকে নিয়ে খারাপ লাগে?", "২৪. মনোযোগে সমস্যা?",
    "২৫. খুব ধীর/খুব দ্রুত নড়াচড়া?", "২৬. নিজেকে আঘাত করার চিন্তা?"
]

# -----------------------------
# HELPERS
# -----------------------------
def extract_number(text):
    if not text:
        return 0.0
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
        return float(match.group()) if match else 0.0
    except:
        return 0.0

def load_resources():
    # If you deploy, these must exist in repo (same folder as app.py)
    try:
        model = joblib.load("mental_health_hybrid_model.pkl")
        encoders = joblib.load("label_encoders.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, encoders, feature_columns
    except Exception as e:
        return None, None, None, str(e)

def is_low_risk_label(label: str) -> bool:
    low_exact = {
        "Minimal Anxiety", "Low Stress", "No Depression", "Minimal Depression",
        "Normal", "None"
    }
    # Accept common “low/minimal/no depression” patterns safely
    return (label in low_exact) or any(x in label for x in ["Minimal", "Low", "No Depression", "No Stress", "No Anxiety"])

def severity_bucket(label: str) -> str:
    # mild / moderate / severe-ish from label text
    if any(x in label for x in ["Severe", "High"]):
        return "Severe/High"
    if "Moderate" in label:
        return "Moderate"
    if any(x in label for x in ["Mild", "Low", "Minimal", "None", "No Depression"]):
        return "Mild"
    return "Mild"

def friendly_tips(condition: str, bucket: str, lang: str):
    # Family-friendly + real-life suggestions; only severe triggers professional help emphasis
    tips_en = {
        "Anxiety": {
            "Mild": ["Take 5 slow breaths and drink water.", "Reduce caffeine today.", "Talk to a friend for 5 minutes."],
            "Moderate": ["Write worries on paper and close it.", "10–15 min walk.", "Keep phone away 30–60 min before sleep."],
            "Severe/High": ["Please talk to a counselor/psychologist.", "Do not stay alone—reach out to family/friends now.", "If unsafe, call emergency/helpline immediately."]
        },
        "Stress": {
            "Mild": ["Do one small task now (not everything).", "Take a short break (5–10 min).", "Eat a normal meal and hydrate."],
            "Moderate": ["Use a simple to-do list (Top 3 tasks).", "Light stretching.", "Ask someone for help instead of carrying it alone."],
            "Severe/High": ["High stress can harm health—seek support from advisor/counselor.", "Take rest urgently.", "If feeling overwhelmed, contact a professional."]
        },
        "Depression": {
            "Mild": ["Open curtains / get sunlight for 10 minutes.", "Do a small cleaning (desk/bed).", "Send a message to someone you trust."],
            "Moderate": ["Short walk outside.", "Try a small hobby (music/game/art).", "Keep a simple routine for today."],
            "Severe/High": ["Please see a psychologist/counselor.", "Tell a family member or trusted person now.", "If self-harm thoughts exist, call helpline/emergency immediately."]
        },
    }
    tips_bn = {
        "Anxiety": {
            "Mild": ["৫ বার ধীরে শ্বাস নিন ও পানি পান করুন।", "আজ ক্যাফেইন কমান।", "কাউকে বিশ্বাস করেন এমন বন্ধুর সাথে ৫ মিনিট কথা বলুন।"],
            "Moderate": ["দুশ্চিন্তাগুলো লিখে খাতা বন্ধ করে রাখুন।", "১০–১৫ মিনিট হাঁটুন।", "ঘুমানোর আগে ৩০–৬০ মিনিট ফোন দূরে রাখুন।"],
            "Severe/High": ["দয়া করে কাউন্সিলর/সাইকোলজিস্টের সাথে কথা বলুন।", "একাই থাকবেন না—পরিবার/বন্ধুকে জানান।", "ঝুঁকি মনে হলে সাথে সাথে হেল্পলাইন/ইমার্জেন্সিতে কল করুন।"]
        },
        "Stress": {
            "Mild": ["একবারে সব না—আজ শুধু ১টা ছোট কাজ করুন।", "৫–১০ মিনিট ব্রেক নিন।", "ঠিকমতো খাওয়া-দাওয়া ও পানি পান করুন।"],
            "Moderate": ["Top-3 টু-ডু লিস্ট করুন।", "হালকা স্ট্রেচিং করুন।", "কারো সাহায্য নিন—একাই সব চাপ নেবেন না।"],
            "Severe/High": ["অতিরিক্ত স্ট্রেসে স্বাস্থ্য ক্ষতি হয়—অ্যাডভাইজার/কাউন্সিলরকে জানান।", "বিশ্রাম জরুরি।", "খুব খারাপ লাগলে পেশাদার সাহায্য নিন।"]
        },
        "Depression": {
            "Mild": ["পর্দা খুলে ১০ মিনিট রোদ নিন।", "টেবিল/বিছানা একটু গুছান।", "বিশ্বাসভাজন কাউকে মেসেজ দিন।"],
            "Moderate": ["বাইরে একটু হাঁটুন।", "ছোট একটা শখের কাজ করুন (গান/গেম/ড্রইং)।", "আজকের জন্য ছোট রুটিন ঠিক করুন।"],
            "Severe/High": ["দয়া করে সাইকোলজিস্ট/কাউন্সিলর দেখান।", "পরিবারের কাউকে এখনই জানান।", "আত্মহানির চিন্তা থাকলে সাথে সাথে হেল্পলাইন/ইমার্জেন্সি কল করুন।"]
        },
    }
    dataset = tips_bn if lang == "Bangla" else tips_en
    return dataset.get(condition, {}).get(bucket, dataset.get(condition, {}).get("Mild", []))

# -----------------------------
# SESSION / RESET
# -----------------------------
if "profile_locked" not in st.session_state:
    st.session_state.profile_locked = False

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.profile_locked = False
    st.rerun()

# -----------------------------
# LANGUAGE
# -----------------------------
st.sidebar.markdown("### 🌐 Language / ভাষা")
lang = st.sidebar.radio("", ("English", "Bangla"), label_visibility="collapsed")
t = translations[lang]

# -----------------------------
# HEADER
# -----------------------------
c1, c2 = st.columns([8, 2])
with c1:
    st.title(t["title"])
    st.caption(t["subtitle"])
with c2:
    if st.button(t["reset_btn"], type="primary"):
        reset_all()

st.markdown("---")

# -----------------------------
# LOAD MODEL
# -----------------------------
model, encoders, feature_columns, err = load_resources()
if model is None:
    st.error("🚨 Model files missing or cannot be loaded.")
    st.code(f"Load error: {err}")
    st.info("Make sure these are in the same folder as app.py: mental_health_hybrid_model.pkl, label_encoders.pkl, feature_columns.pkl")
    st.stop()

# -----------------------------
# PROFILE (STRICT GATE + LOCK)
# -----------------------------
st.sidebar.header(t["sidebar_title"])

with st.sidebar.form("profile_form", clear_on_submit=False):
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

# Validate profile
sentinel_values = {t["select"], "Select...", "সিলেক্ট করুন..."}  # just extra safety
def is_selected(x): 
    return x and (x not in sentinel_values) and (not str(x).startswith("Select")) and (not str(x).startswith("সিলেক্ট"))

profile_complete = all([
    student_name and student_name.strip(),
    is_selected(age_input),
    is_selected(gender_input),
    is_selected(uni_input),
    is_selected(dept_input),
    is_selected(year_input),
    is_selected(sch_input),
    cgpa_input > 0.0,
    confirm_ok
])

if lock_btn:
    if profile_complete:
        st.session_state.profile_locked = True
        st.sidebar.success("✅ Profile saved!" if lang == "English" else "✅ প্রোফাইল সেভ হয়েছে!")
    else:
        st.session_state.profile_locked = False
        st.sidebar.error("Please complete all fields and confirm." if lang == "English" else "সব তথ্য পূরণ করে কনফার্ম করুন।")

# Helpline
with st.sidebar.expander(t["helpline_title"], expanded=True):
    st.markdown("""
📞 **Kaan Pete Roi:** 01779554391  
📞 **Moner Bondhu:** 01779632588  
🚑 **National Emergency:** 999
""")

# Gate questionnaire
if not st.session_state.profile_locked:
    st.warning(t["fill_profile_msg"])
    st.markdown(f"<div class='locked-hint'>👈 {'Complete the sidebar profile first.' if lang=='English' else 'বাম পাশের প্রোফাইল আগে পূরণ করুন।'}</div>", unsafe_allow_html=True)
    st.stop()

# -----------------------------
# MAP TO MODEL VALUES (AFTER LOCK)
# -----------------------------
gender_model = "Male" if gender_input in ["Male", "পুরুষ"] else "Female"
uni_model = "Public" if uni_input in ["Public", "পাবলিক"] else "Private"
sch_model = "Yes" if sch_input in ["Yes", "হ্যাঁ"] else "No"

dept_map = {"সিএসই": "CSE", "ইইই": "EEE", "বিবিএ": "BBA", "ইংরেজি": "English", "আইন": "Law", "ফার্মাসি": "Pharmacy", "অন্যান্য": "Other"}
dept_model = dept_map.get(dept_input, dept_input)

year_map = {"১ম বর্ষ": "First Year", "২য় বর্ষ": "Second Year", "৩য় বর্ষ": "Third Year", "৪র্থ বর্ষ": "Fourth Year", "মাস্টার্স": "Master"}
year_model = year_map.get(year_input, year_input)

# -----------------------------
# QUESTIONNAIRE (RADIO OPTIONS)
# -----------------------------
st.subheader(("👋 Hello, " if lang == "English" else "👋 হ্যালো, ") + student_name.strip())
st.subheader(t["section_title"])
st.info(t["instructions"])

radio_options = t["radio_opts"]
options_map = {
    "Not at all": 0, "একদম না": 0,
    "Sometimes": 1, "মাঝে মাঝে": 1,
    "Often": 2, "প্রায়ই": 2,
    "Very Often": 3, "খুব বেশি": 3
}

q_labels = q_labels_bn if lang == "Bangla" else q_labels_en
answers = []

# Use a form so user doesn’t trigger reruns per question
with st.form("questions_form"):
    colL, colR = st.columns(2)
    for i, q in enumerate(q_labels):
        with (colL if i % 2 == 0 else colR):
            val = st.radio(f"**{q}**", radio_options, horizontal=True, key=f"q_{i}")
            answers.append(options_map[val])
            st.divider()
    analyze_btn = st.form_submit_button(t["analyze_btn"], type="primary", use_container_width=True)

# -----------------------------
# PREDICTION + FRIENDLY OUTPUT
# -----------------------------
if analyze_btn:
    age_numeric = extract_number(age_input)
    cgpa_numeric = float(cgpa_input)

    if not (isinstance(feature_columns, (list, tuple)) and len(feature_columns) == 33):
        st.error("Feature column mismatch (expected 33).")
        st.stop()

    input_dict = {
        feature_columns[0]: age_numeric,
        feature_columns[1]: gender_model,
        feature_columns[2]: uni_model,
        feature_columns[3]: dept_model,
        feature_columns[4]: year_model,
        feature_columns[5]: cgpa_numeric,
        feature_columns[6]: sch_model,
    }
    for i in range(26):
        input_dict[feature_columns[7 + i]] = answers[i]

    input_df = pd.DataFrame([input_dict]).reindex(columns=feature_columns, fill_value=0)

    with st.spinner(t["analyzing"]):
        probs = model.predict_proba(input_df)

    # Global safety alert for Q26
    if answers[25] >= 2:
        st.markdown(f"""
        <div class="emergency-box">
            <h3>🚨 {"Emergency Alert" if lang=="English" else "জরুরি সতর্কতা"}</h3>
            <p>{t["emergency_text"]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.success(t["success"])
    st.subheader(t["result_title"])

    conditions = ["Anxiety", "Stress", "Depression"]
    cards = st.columns(3)

    report_lines = []
    report_lines.append("--- WELLNESS REPORT ---")
    report_lines.append(f"Name: {student_name.strip()}")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Profile: Age={age_input}, Gender={gender_model}, Dept={dept_model}, CGPA={cgpa_numeric:.2f}")
    report_lines.append("----------------------")

    risk_list = []  # (cond, confidence, label, bucket, low_risk)

    for i, cond in enumerate(conditions):
        prob_arr = probs[i][0]
        best_idx = int(np.argmax(prob_arr))
        label = encoders[f"{cond} Label"].inverse_transform([best_idx])[0]
        confidence = float(prob_arr[best_idx]) * 100.0

        low_risk = is_low_risk_label(label)
        bucket = severity_bucket(label)

        # Display label (simple and clear)
        if lang == "Bangla":
            if low_risk:
                display_label = "ঝুঁকি নেই / কম"
            elif "Moderate" in label:
                display_label = "মাঝারি"
            elif any(x in label for x in ["Severe", "High"]):
                display_label = "তীব্র"
            else:
                display_label = "মৃদু"
        else:
            if low_risk:
                display_label = f"No/Low {cond}"
            else:
                display_label = label

        with cards[i]:
            st.markdown(f"### {cond}")
            if low_risk:
                st.success(f"**{display_label}**")
                st.progress(0)
                # Clinical note inside depression card if Q26 high
                if cond == "Depression" and answers[25] >= 2:
                    st.warning(t["clinical_note"])
            else:
                st.error(f"**{display_label}**")
                st.progress(min(100, max(1, int(confidence))))
                st.caption((f"Risk confidence: {confidence:.1f}%" if lang == "English" else f"ঝুঁকির কনফিডেন্স: {confidence:.1f}%"))

        report_lines.append(f"{cond}: {label} (Confidence {confidence:.1f}%)")
        risk_list.append((cond, confidence, label, bucket, low_risk))

    st.markdown("---")
    st.subheader(t["suggestions"])

    # Friendly suggestions: show top concerns first, but only where not low risk
    concerns = [x for x in risk_list if not x[4]]
    concerns.sort(key=lambda x: x[1], reverse=True)

    if not concerns:
        st.markdown(f"<div class='suggestion-friendly'>{t['healthy_msg']}</div>", unsafe_allow_html=True)
        report_lines.append("Tips: Maintain routine, sleep, hydration, and social connection.")
    else:
        for cond, conf, label, bucket, _ in concerns:
            tips = friendly_tips(cond, bucket, lang)
            serious = (bucket == "Severe/High") or (cond == "Depression" and answers[25] >= 2)
            box_class = "suggestion-serious" if serious else "suggestion-friendly"

            title = (f"👉 {cond} Tips" if lang == "English" else f"👉 {cond} পরামর্শ")
            st.markdown(f"#### {title}")
            st.markdown(
                f"<div class='{box_class}'><ul style='margin:0; padding-left:18px;'>"
                + "".join([f"<li>{t}</li>" for t in tips])
                + "</ul></div>",
                unsafe_allow_html=True
            )

            report_lines.append(f"[{cond} Tips - {bucket}]")
            report_lines.extend([f"- {re.sub(r'[*_`]', '', tip)}" for tip in tips])

    st.markdown("---")
    report_lines.append("----------------------")
    report_lines.append("DISCLAIMER: This is ML-based screening and not a clinical diagnosis.")

    st.download_button(
        label=t["download_btn"],
        data="\n".join(report_lines),
        file_name=f"Wellness_Report_{student_name.strip().replace(' ', '_')}.txt",
        mime="text/plain"
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"<div class='footer'>{t['dev_by']} | {t['disclaimer_short']}</div>", unsafe_allow_html=True)
