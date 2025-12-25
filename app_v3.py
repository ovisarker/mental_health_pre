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

# Custom CSS for Friendly UI
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
    /* Friendly Suggestion Styling */
    .suggestion-friendly {
        background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 5px solid #2196f3; margin-top: 10px; color: #0d47a1;
    }
    .suggestion-serious {
        background-color: #fff3e0; padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800; margin-top: 10px; color: #e65100;
    }
    /* Blur effect for locked state */
    .locked-content {
        filter: blur(5px);
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# --- TRANSLATION DICTIONARY ---
translations = {
    'English': {
        'title': "Student Mental Health Assessment",
        'subtitle': "Machine Learning Based Screening System",
        'reset_btn': "🔄 Reset System",
        'sidebar_title': "📝 Student Profile (Required)",
        'name': "Student Name",
        'age': "1. Age Group",
        'gender': "2. Gender",
        'uni': "3. University Type",
        'dept': "4. Department",
        'year': "5. Academic Year",
        'cgpa': "6. Current CGPA",
        'scholarship': "7. Scholarship/Waiver?",
        'helpline_title': "🆘 Emergency Helpline (BD)",
        'section_title': "📋 Behavioral Assessment",
        'instructions': "💡 **Instructions:** Select the option that best describes your feelings **over the last 2 weeks**.",
        'analyze_btn': "🚀 Analyze My Mental Health",
        'analyzing': "Analyzing behavioral patterns...",
        'success': "✅ Assessment Complete",
        'result_title': "📊 Your Wellness Report",
        'suggestions': "💡 Friendly Suggestions",
        'healthy_msg': "🎉 **You are doing great!**\nYour responses show you are mentally strong right now. Keep enjoying your life!",
        'download_btn': "📥 Download Report",
        'disclaimer_short': "⚠️ **Note:** This is an ML-based tool for screening, not a doctor replacement.",
        'dev_by': "Developed by",
        'fill_profile_msg': "🚫 **Action Required:** Please fill up the **Student Profile** on the left sidebar completely to unlock the questions.",
        'radio_opts': ["Not at all", "Sometimes", "Often", "Very Often"],
        'genders': ['Male', 'Female'],
        'unis': ['Public', 'Private'],
        'scholars': ['Yes', 'No'],
        'years': ['First Year', 'Second Year', 'Third Year', 'Fourth Year', 'Master'],
        'depts': ["CSE", "EEE", "BBA", "English", "Law", "Pharmacy", "Other"]
    },
    'Bangla': {
        'title': "শিক্ষার্থী মানসিক স্বাস্থ্য মূল্যায়ন",
        'subtitle': "মেশিন লার্নিং ভিত্তিক স্ক্রিনিং সিস্টেম",
        'reset_btn': "🔄 রিসেট করুন",
        'sidebar_title': "📝 প্রোফাইল (অবশ্যই পূরণীয়)",
        'name': "শিক্ষার্থীর নাম",
        'age': "১. বয়স গ্রুপ",
        'gender': "২. লিঙ্গ",
        'uni': "৩. বিশ্ববিদ্যালয়ের ধরণ",
        'dept': "৪. ডিপার্টমেন্ট",
        'year': "৫. শিক্ষাবর্ষ",
        'cgpa': "৬. বর্তমান সিজিপিএ (CGPA)",
        'scholarship': "৭. স্কলারশিপ/ওয়েভার আছে?",
        'helpline_title': "🆘 জরুরি হেল্পলাইন (BD)",
        'section_title': "📋 আচরণগত মূল্যায়ন",
        'instructions': "💡 **নির্দেশনা:** গত **২ সপ্তাহে** আপনার অনুভূতির সাথে যে অপশনটি মিলে, সেটি সিলেক্ট করুন।",
        'analyze_btn': "🚀 ফলাফল দেখুন",
        'analyzing': "বিশ্লেষণ করা হচ্ছে...",
        'success': "✅ মূল্যায়ন সম্পন্ন",
        'result_title': "📊 আপনার ওয়েলনেস রিপোর্ট",
        'suggestions': "💡 বন্ধুত্বপূর্ণ পরামর্শ",
        'healthy_msg': "🎉 **আপনি দারুণ আছেন!**\nআপনার মানসিক অবস্থা বেশ ভালো। জীবন উপভোগ করতে থাকুন!",
        'download_btn': "📥 রিপোর্ট ডাউনলোড",
        'disclaimer_short': "⚠️ **নোট:** এটি একটি স্ক্রিনিং টুল, ডাক্তারের বিকল্প নয়।",
        'dev_by': "ডেভেলপ করেছে",
        'fill_profile_msg': "🚫 **অ্যাকশন প্রয়োজন:** প্রশ্নগুলো দেখার জন্য দয়া করে বাম পাশের **শিক্ষার্থীর প্রোফাইল** সম্পূর্ণ পূরণ করুন।",
        'radio_opts': ["একদম না", "মাঝে মাঝে", "প্রায়ই", "খুব বেশি"],
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

# --- RESET LOGIC ---
if 'reset' not in st.session_state:
    st.session_state.reset = False

def reset_app():
    st.session_state.reset = True
    st.rerun()

# --- LOAD RESOURCES ---
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

def extract_number(text):
    try:
        if pd.isna(text): return 0.0
        text_str = str(text)
        if '-' in text_str: return float(text_str.split('-')[0].strip())
        match = re.search(r"[-+]?\d*\.\d+|\d+", text_str)
        return float(match.group()) if match else 0.0
    except: return 0.0

# --- FRIENDLY vs SERIOUS SUGGESTIONS ---
def get_casual_suggestions(condition, severity, lang):
    # Friendly Tone for Mild/Moderate, Serious for Severe
    
    suggestions_en = {
        "Anxiety": {
            "Mild": ["👋 **Hey!** Just take a deep breath. Listen to your favorite song.", "☕ **Chill:** Skip that extra cup of coffee today.", "🌿 **Nature:** Go out, look at the sky for 5 mins."],
            "Moderate": ["📓 **Write it out:** Got worries? Write them down and tear the paper.", "🏃 **Move:** Do a quick 10-min dance or walk.", "📵 **Disconnect:** Stay away from phone before sleep."],
            "Severe/High": ["🩺 **Doctor's Advice:** It seems heavy. Please see a university counselor.", "🆘 **Help:** Don't fight this alone. Talk to a professional.", "💊 **Health:** Your mental peace is priority. Seek help."]
        },
        "Stress": {
            "Mild": ["🎬 **Movie Time:** Watch something funny today.", "📅 **One thing:** Just do one small task, don't overthink.", "🍕 **Treat:** Eat something you love!"],
            "Moderate": ["🧘 **Relax:** Try stretching or yoga.", "🚫 **Say No:** Don't take extra pressure.", "🗣️ **Share:** Rant to a friend, it helps."],
            "Severe/High": ["🩺 **Professional Help:** You might be burning out. Talk to an advisor.", "🛑 **Stop:** Take a break before you crash.", "🆘 **Support:** High stress hurts health. Consult a doctor."]
        },
        "Depression": {
            "Mild": ["👋 **Friend:** Call your best friend today.", "🌞 **Sun:** Go stand in the sun for 10 mins.", "🧹 **Tidy:** Just clean your desk, it feels good."],
            "Moderate": ["🎨 **Hobby:** Do something you used to love (drawing/gaming).", "🥗 **Food:** Eat a good meal.", "🚶 **Walk:** Just a short walk outside."],
            "Severe/High": ["🩺 **Urgent:** Please visit a psychologist or counselor.", "🆘 **Helpline:** If you feel unsafe, call the helpline below.", "🤝 **Family:** Tell your family how you feel."]
        }
    }

    suggestions_bn = {
        "Anxiety": {
            "Mild": ["👋 **হেই!** লম্বা শ্বাস নাও। পছন্দের গানটা শোনো।", "☕ **চিল:** আজকে আর কফি খেও না।", "🌿 **প্রকৃতি:** বাইরে গিয়ে ৫ মিনিট আকাশ দেখো।"],
            "Moderate": ["📓 **লিখে ফেলো:** দুশ্চিন্তা হচ্ছে? লিখে কাগজটা ছিঁড়ে ফেলো।", "🏃 **নড়াচড়া:** ১০ মিনিট একটু হাঁটো বা নাচো।", "📵 **ফোন দূরে:** ঘুমানোর আগে ফোন টিপবে না।"],
            "Severe/High": ["🩺 **পরামর্শ:** এটা একটু বেশি মনে হচ্ছে। দয়া করে কাউন্সিলর দেখাও।", "🆘 **সাহায্য:** একা কষ্ট পেও না। ডাক্তার বা প্রফেশনালের সাথে কথা বলো।", "💊 **স্বাস্থ্য:** তোমার শান্তি সবার আগে। সাহায্য নাও।"]
        },
        "Stress": {
            "Mild": ["🎬 **মুভি টাইম:** আজকে মজার কিছু দেখো।", "📅 **ছোট কাজ:** বেশি ভেবো না, ছোট একটা কাজ শেষ করো।", "🍕 **খাবার:** পছন্দের খাবার খাও!"],
            "Moderate": ["🧘 **রিলাক্স:** একটু স্ট্রেচিং বা ইয়োগা করো।", "🚫 **না বলো:** অতিরিক্ত প্রেশার নিও না।", "🗣️ **কথা বলো:** বন্ধুর সাথে মন খুলে কথা বলো।"],
            "Severe/High": ["🩺 **পেশাদার সাহায্য:** তুমি বার্ন-আউট হয়ে যাচ্ছো। অ্যাডভাইজারের সাথে কথা বলো।", "🛑 **থামো:** শরীর খারাপ হওয়ার আগেই ব্রেক নাও।", "🆘 **সাপোর্ট:** অতিরিক্ত স্ট্রেস ক্ষতিকর। ডাক্তার দেখাও।"]
        },
        "Depression": {
            "Mild": ["👋 **বন্ধু:** বেস্ট ফ্রেন্ডকে একটা কল দাও।", "🌞 **রোদ:** সকালে ১০ মিনিট রোদে দাঁড়িয়ে থাকো।", "🧹 **গোছানো:** পড়ার টেবিলটা একটু গুছিয়ে নাও, ভালো লাগবে।"],
            "Moderate": ["🎨 **শখ:** ছবি আঁকা বা গেম খেলা—যা ভালো লাগে তাই করো।", "🥗 **খাবার:** ঠিকমতো খাওয়া-দাওয়া করো।", "🚶 **হাঁটা:** বিকেলে একটু বাইরে হেঁটে এসো।"],
            "Severe/High": ["🩺 **জরুরি:** দয়া করে সাইকোলজিস্ট বা কাউন্সিলর দেখাও।", "🆘 **হেল্পলাইন:** যদি খুব খারাপ লাগে, হেল্পলাইনে কল করো।", "🤝 **পরিবার:** পরিবারের কাউকে তোমার অনুভূতির কথা জানাও।"]
        }
    }

    dataset = suggestions_bn if lang == 'Bangla' else suggestions_en
    level_key = "Mild"
    if "Moderate" in severity: level_key = "Moderate"
    elif "Severe" in severity or "High" in severity: level_key = "Severe/High"
    
    return dataset.get(condition, {}).get(level_key, dataset[condition]["Mild"])

# --- LANGUAGE SELECTOR ---
st.sidebar.markdown("### 🌐 Language / ভাষা")
lang = st.sidebar.radio("", ('English', 'Bangla'), label_visibility="collapsed")
t = translations[lang]

# --- HEADER ---
col1, col2 = st.columns([8, 2])
with col1:
    st.title(t['title'])
    st.markdown(f"##### {t['subtitle']}")
with col2:
    if st.button(t['reset_btn'], type="primary"):
        reset_app()

st.markdown("---")

# --- SIDEBAR PROFILE (STRICT MODE) ---
st.sidebar.header(t['sidebar_title'])

# 1. Name Input (NEW & REQUIRED)
student_name = st.sidebar.text_input(t['name'], placeholder="Enter full name")

# 2. Dynamic Inputs (Forced Selection via Index=None)
# We use index=None to force user to pick one. If not picked, it stays None.

age_display = ['18-22', '23-26', '27-30', 'Above 30']
age_input = st.sidebar.selectbox(t['age'], age_display, index=None, placeholder="Select Age")

gender_input = st.sidebar.selectbox(t['gender'], t['genders'], index=None, placeholder="Select Gender")

uni_input = st.sidebar.selectbox(t['uni'], t['unis'], index=None, placeholder="Select Type")

dept_input = st.sidebar.selectbox(t['dept'], t['depts'], index=None, placeholder="Select Dept")

year_input = st.sidebar.selectbox(t['year'], t['years'], index=None, placeholder="Select Year")

cgpa_input = st.sidebar.number_input(t['cgpa'], min_value=0.00, max_value=4.00, value=0.00, step=0.01, format="%.2f")

sch_input = st.sidebar.selectbox(t['scholarship'], t['scholars'], index=None, placeholder="Select Option")

st.sidebar.markdown("---")

# Helpline
with st.sidebar.expander(t['helpline_title'], expanded=True):
    st.markdown("""
    📞 **Kaan Pete Roi:** 01779554391  
    📞 **Moner Bondhu:** 01779632588  
    🚑 **National Emergency:** 999
    """)

# --- STRICT VALIDATION LOGIC ---
# Check if all fields are filled
is_profile_complete = all([
    student_name, 
    age_input, 
    gender_input, 
    uni_input, 
    dept_input, 
    year_input, 
    sch_input,
    cgpa_input > 0.0  # Ensure CGPA is entered
])

if not is_profile_complete:
    st.warning(t['fill_profile_msg'])
    st.info("👈 Please check the sidebar.")
    st.stop() # Stops execution here until profile is filled

# --- MAPPING FOR MODEL (After Validation) ---
gender_model = 'Male' if gender_input in ['Male', 'পুরুষ'] else 'Female'
uni_model = 'Public' if uni_input in ['Public', 'পাবলিক'] else 'Private'
sch_model = 'Yes' if sch_input in ['Yes', 'হ্যাঁ'] else 'No'

# Dept Map
dept_map = {"সিএসই": "CSE", "ইইই": "EEE", "বিবিএ": "BBA", "ইংরেজি": "English", "আইন": "Law", "ফার্মাসি": "Pharmacy", "অন্যান্য": "Other"}
dept_model = dept_map.get(dept_input, dept_input)

# Year Map
year_map = {'১ম বর্ষ': 'First Year', '২য় বর্ষ': 'Second Year', '৩য় বর্ষ': 'Third Year', '৪র্থ বর্ষ': 'Fourth Year', 'মাস্টার্স': 'Master'}
year_model = year_map.get(year_input, year_input)

# --- QUESTIONNAIRE (RADIO BUTTONS) ---

st.subheader(f"👋 Hello, {student_name}")
st.subheader(t['section_title'])
st.info(t['instructions'])

radio_options = t['radio_opts']
# Standardized Mapping
options_map = {
    "Not at all": 0, "একদম না": 0,
    "Sometimes": 1, "মাঝে মাঝে": 1,
    "Often": 2, "প্রায়ই": 2,
    "Very Often": 3, "খুব বেশি": 3
}

q_labels = q_labels_bn if lang == 'Bangla' else q_labels_en
answers_map = {}
q_col1, q_col2 = st.columns(2)

for i, q_text in enumerate(q_labels):
    current_col = q_col1 if i % 2 == 0 else q_col2
    with current_col:
        # RADIO BUTTON (Horizontal)
        key_name = f"q_{i}_{st.session_state.reset}_{lang}"
        val = st.radio(label=f"**{q_text}**", options=radio_options, horizontal=True, key=key_name)
        answers_map[i] = options_map[val]
        st.write("---") # Separator line

final_answers = [answers_map[i] for i in range(26)]

if st.session_state.reset:
    st.session_state.reset = False

# --- PREDICTION ---
col_cen1, col_cen2, col_cen3 = st.columns([1, 2, 1])
with col_cen2:
    analyze_btn = st.button(t['analyze_btn'], type="primary", use_container_width=True)

if analyze_btn:
    age_numeric = extract_number(age_input)
    cgpa_numeric = float(cgpa_input)
    
    input_dict = {}
    if len(feature_columns) == 33:
        input_dict[feature_columns[0]] = age_numeric
        input_dict[feature_columns[1]] = gender_model
        input_dict[feature_columns[2]] = uni_model
        input_dict[feature_columns[3]] = dept_model
        input_dict[feature_columns[4]] = year_model
        input_dict[feature_columns[5]] = cgpa_numeric
        input_dict[feature_columns[6]] = sch_model
        for i in range(26):
            input_dict[feature_columns[7+i]] = final_answers[i]
            
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        try:
            with st.spinner(t['analyzing']):
                probs = model.predict_proba(input_df)
            
            # --- GLOBAL SAFETY ALERT ---
            if final_answers[25] >= 2:
                st.markdown(f"""
                <div class="emergency-box">
                    <h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3>
                    <p>{'Please seek professional help immediately.' if lang=='English' else 'দয়া করে অবিলম্বে পেশাদার সাহায্য নিন।'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success(t['success'])
            st.subheader(t['result_title'])
            
            result_cols = st.columns(3)
            conditions = ['Anxiety', 'Stress', 'Depression']
            risk_scores = []
            healthy_count = 0
            
            report_text = f"--- WELLNESS REPORT ---\n"
            report_text += f"Name: {student_name}\n"
            report_text += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            report_text += "-----------------------\n\n"
            
            for i, cond in enumerate(conditions):
                prob_arr = probs[i][0]
                best_idx = np.argmax(prob_arr)
                label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                confidence = prob_arr[best_idx] * 100
                
                # Check low risk
                low_set = {"Minimal Anxiety", "Low Stress", "No Depression", "Minimal Depression", "Normal", "None"}
                is_healthy = label in low_set or any(x in label for x in ["Minimal", "Low", "No Depression"])
                
                # Translate Display Label
                display_label = label
                if lang == 'Bangla':
                    if is_healthy: display_label = "ঝুঁকি নেই / সুস্থ"
                    elif "Severe" in label: display_label = "তীব্র ঝুঁকি"
                    elif "Moderate" in label: display_label = "মাঝারি ঝুঁকি"
                    elif "Mild" in label: display_label = "মৃদু ঝুঁকি"
                else:
                    if is_healthy: display_label = f"No/Low {cond}"

                report_text += f"{cond}: {label}\n"
                
                with result_cols[i]:
                    st.markdown(f"#### {cond}")
                    if is_healthy:
                        st.success(f"**{display_label}**")
                        st.progress(0)
                        healthy_count += 1
                        risk_scores.append((cond, 0, label))
                        
                        if cond == 'Depression' and final_answers[25] >= 2:
                             st.warning("⚠️ **Note:** Self-harm risk detected.")
                    else:
                        st.error(f"**{display_label}**")
                        st.progress(int(confidence))
                        risk_scores.append((cond, confidence, label))

            st.markdown("---")
            
            # --- FRIENDLY SUGGESTIONS ---
            st.subheader(t['suggestions'])
            
            if healthy_count == 3:
                st.balloons()
                st.markdown(f"""
                <div class="suggestion-friendly">
                    {t['healthy_msg']}
                </div>
                """, unsafe_allow_html=True)
                report_text += "\nResult: Healthy & Balanced."
            else:
                risk_scores.sort(key=lambda x: x[1], reverse=True)
                for cond, conf, severity_label in risk_scores:
                    if conf > 0:
                        is_severe = "Severe" in severity_label or "High" in severity_label
                        box_class = "suggestion-serious" if is_severe else "suggestion-friendly"
                        
                        st.markdown(f"##### 👉 **{cond}**")
                        suggestions = get_casual_suggestions(cond, severity_label, lang)
                        
                        # Render suggestion box
                        html_content = f"""
                        <div class="{box_class}">
                            <ul style="margin:0; padding-left:20px;">
                                {''.join([f'<li>{s}</li>' for s in suggestions])}
                            </ul>
                        </div>
                        """
                        st.markdown(html_content, unsafe_allow_html=True)
                        
                        # Add to report
                        clean_tips = [s.replace('**', '').replace('👋', '').strip() for s in suggestions]
                        report_text += f"\n[{cond} Tips]:\n" + "\n".join(clean_tips) + "\n"

            st.markdown("---")
            
            st.download_button(
                label=t['download_btn'],
                data=report_text,
                file_name=f"Wellness_Report_{student_name}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Model Error")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown(f"""<div class='footer'>{t['dev_by']} <strong>Team Dual Core</strong> | {t['disclaimer_short']}</div>""", unsafe_allow_html=True)
