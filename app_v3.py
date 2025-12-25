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

# Custom CSS for Professional UI
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
</style>
""", unsafe_allow_html=True)

# --- TRANSLATION DICTIONARY ---
translations = {
    'English': {
        'title': "Student Mental Health Assessment & Risk Prediction",
        'subtitle': "Machine Learning Based Screening System",
        'reset_btn': "🔄 Reset Assessment",
        'sidebar_title': "📝 Student Profile",
        'age': "1. Age Group",
        'gender': "2. Gender",
        'uni': "3. University Type",
        'dept': "4. Department",
        'year': "5. Academic Year",
        'cgpa': "6. Current CGPA",
        'scholarship': "7. Scholarship/Waiver?",
        'helpline_title': "🆘 Emergency Helpline (BD)",
        'section_title': "📋 Behavioral Assessment",
        'instructions': "💡 **Instructions:** Please indicate how frequently you have experienced these feelings **over the last 2 weeks**.",
        'scale_caption': "Scale: **Not at all** (0) → **Sometimes** (1) → **Often** (2) → **Very Often** (3)",
        'analyze_btn': "🚀 Analyze Risk Level",
        'analyzing': "Analyzing behavioral patterns...",
        'success': "✅ Assessment Complete",
        'result_title': "📊 Clinical Risk Assessment",
        'suggestions': "💡 Professional Recommendations",
        'healthy_msg': "🎉 **Status: Mentally Resilient**\nYour responses indicate a stable mental state. Continue maintaining your current lifestyle.",
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
        'title': "শিক্ষার্থী মানসিক স্বাস্থ্য ও ঝুঁকি মূল্যায়ন",
        'subtitle': "মেশিন লার্নিং ভিত্তিক স্ক্রিনিং সিস্টেম",
        'reset_btn': "🔄 রিসেট করুন",
        'sidebar_title': "📝 শিক্ষার্থীর প্রোফাইল",
        'age': "১. বয়স গ্রুপ",
        'gender': "২. লিঙ্গ",
        'uni': "৩. বিশ্ববিদ্যালয়ের ধরণ",
        'dept': "৪. ডিপার্টমেন্ট",
        'year': "৫. শিক্ষাবর্ষ",
        'cgpa': "৬. বর্তমান সিজিপিএ (CGPA)",
        'scholarship': "৭. স্কলারশিপ/ওয়েভার আছে?",
        'helpline_title': "🆘 জরুরি হেল্পলাইন (BD)",
        'section_title': "📋 আচরণগত মূল্যায়ন",
        'instructions': "💡 **নির্দেশনা:** গত **২ সপ্তাহে** আপনি এই অনুভূতিগুলো কতবার অনুভব করেছেন তা জানান।",
        'scale_caption': "স্কেল: **একদম না** (০) → **মাঝে মাঝে** (১) → **প্রায়ই** (২) → **খুব বেশি** (৩)",
        'analyze_btn': "🚀 ঝুঁকি বিশ্লেষণ করুন",
        'analyzing': "মেশিন লার্নিং মডেল বিশ্লেষণ করছে...",
        'success': "✅ মূল্যায়ন সম্পন্ন হয়েছে",
        'result_title': "📊 ক্লিনিক্যাল ঝুঁকি মূল্যায়ন",
        'suggestions': "💡 পেশাদার পরামর্শ",
        'healthy_msg': "🎉 **অবস্থা: মানসিকভাবে স্থিতিশীল**\nআপনার মানসিক অবস্থা ভারসাম্যপূর্ণ। বর্তমান জীবনধারা বজায় রাখুন।",
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

# --- PROFESSIONAL MEDICAL SUGGESTIONS ---
def get_professional_suggestions(condition, severity, lang):
    suggestions_en = {
        "Anxiety": {
            "Mild": ["**Self-Care:** Practice 'Box Breathing' (4-4-4-4 technique) twice daily.", "**Lifestyle:** Reduce caffeine intake after 2 PM.", "**Mindfulness:** Spend 10 mins in nature."],
            "Moderate": ["**Management:** Start a 'Worry Journal' - write down worries and close the book.", "**Physical:** Engage in 20 mins of aerobic exercise daily.", "**Sleep:** Maintain strict sleep hygiene."],
            "Severe/High": ["**Urgent Action:** Please consult a university counselor or psychologist.", "**Immediate Coping:** Use the 5-4-3-2-1 Grounding Technique immediately.", "**Support:** Reach out to a trusted friend."]
        },
        "Stress": {
            "Mild": ["**Organization:** Use the Eisenhower Matrix to prioritize tasks.", "**Break:** Take a 5-minute break for every 25 minutes of study.", "**Social:** Talk to a friend about non-academic topics."],
            "Moderate": ["**Relaxation:** Practice Progressive Muscle Relaxation (PMR) before sleep.", "**Balance:** Ensure you are not skipping meals or sleep.", "**Activity:** Light yoga or stretching."],
            "Severe/High": ["**Intervention:** Academic load may be unmanageable; speak to your advisor.", "**Health:** High stress affects immunity; prioritize rest.", "**Professional:** Consider stress management counseling."]
        },
        "Depression": {
            "Mild": ["**Routine:** Stick to a small, manageable daily routine.", "**Sunlight:** Get 15-20 minutes of morning sunlight.", "**Connection:** Send a message to one friend today."],
            "Moderate": ["**Activity:** Behavioral Activation - do one thing you used to enjoy.", "**Diet:** Focus on Omega-3 rich foods.", "**Sleep:** Avoid daytime napping."],
            "Severe/High": ["**Critical:** Please contact a mental health professional or the helpline.", "**Safety:** If you have thoughts of self-harm, call the emergency number.", "**Support:** Do not isolate yourself."]
        }
    }

    suggestions_bn = {
        "Anxiety": {
            "Mild": ["**যত্ন:** দিনে দুবার 'বক্স ব্রিদিং' (৪-৪-৪-৪ টেকনিক) অনুশীলন করুন।", "**জীবনধারা:** দুপুর ২টার পর চা/কফি খাওয়া কমিয়ে দিন।", "**মনোযোগ:** প্রকৃতির মাঝে ১০ মিনিট সময় কাটান।"],
            "Moderate": ["**ব্যবস্থাপনা:** 'দুশ্চিন্তার ডায়েরি' লিখুন।", "**ব্যায়াম:** প্রতিদিন ২০ মিনিট অ্যারোবিক ব্যায়াম করুন।", "**ঘুম:** ঘুমানোর ১ ঘণ্টা আগে মোবাইল ব্যবহার বন্ধ রাখুন।"],
            "Severe/High": ["**জরুরি:** দয়া করে বিশ্ববিদ্যালয়ের কাউন্সিলর বা সাইকোলজিস্টের পরামর্শ নিন।", "**তাৎক্ষণিক:** খুব অস্থির লাগলে ৫-৪-৩-২-১ টেকনিক ব্যবহার করুন।", "**সমর্থন:** আজই কোনো বিশ্বস্ত বন্ধুর সাথে কথা বলুন।"]
        },
        "Stress": {
            "Mild": ["**অর্গানাইজেশন:** কাজের গুরুত্ব অনুযায়ী তালিকা (To-Do List) তৈরি করুন।", "**বিরতি:** প্রতি ২৫ মিনিট পড়ার পর ৫ মিনিট বিরতি নিন।", "**সামাজিক:** বন্ধুর সাথে পড়াশোনার বাইরের বিষয় নিয়ে কথা বলুন।"],
            "Moderate": ["**শিথিলকরণ:** ঘুমানোর আগে পেশী শিথিলকরণ (PMR) ব্যায়াম করুন।", "**ভারসাম্য:** পড়ার জন্য খাওয়া বা ঘুম বাদ দিবেন না।", "**ব্যায়াম:** হালকা যোগব্যায়াম স্ট্রেস কমাতে সাহায্য করে।"],
            "Severe/High": ["**হস্তক্ষেপ:** পড়ার চাপ অসহনীয় হলে অ্যাডভাইজারের সাথে কথা বলুন।", "**স্বাস্থ্য:** অতিরিক্ত স্ট্রেস রোগ প্রতিরোধ ক্ষমতা কমায়; বিশ্রামে অগ্রাধিকার দিন।", "**পেশাদার:** স্ট্রেস ম্যানেজমেন্ট কাউন্সিলিংয়ের কথা বিবেচনা করুন।"]
        },
        "Depression": {
            "Mild": ["**রুটিন:** প্রতিদিন ছোট ছোট কাজ করার অভ্যাস করুন।", "**রোদ:** সকালে ১৫-২০ মিনিট গায়ে রোদ লাগান।", "**যোগাযোগ:** আজ অন্তত একজন বন্ধুকে মেসেজ বা কল করুন।"],
            "Moderate": ["**সক্রিয়তা:** ভালো না লাগলেও পছন্দের কাজ করার চেষ্টা করুন।", "**খাবার:** ওমেগা-৩ সমৃদ্ধ খাবার (মাছ, বাদাম) খাওয়ার চেষ্টা করুন।", "**ঘুম:** রাতে ভালো ঘুমের জন্য দিনের বেলা ঘুমানো এড়িয়ে চলুন।"],
            "Severe/High": ["**গুরুত্বপূর্ণ:** দয়া করে মানসিক স্বাস্থ্য বিশেষজ্ঞ বা হেল্পলাইনে যোগাযোগ করুন।", "**নিরাপত্তা:** যদি নিজেকে আঘাত করার চিন্তা আসে, তবে জরুরি নম্বরে কল করুন।", "**সমর্থন:** একা থাকবেন না; বিশ্বাসভাজন কারো সাথে সময় কাটান।"]
        }
    }

    dataset = suggestions_bn if lang == 'Bangla' else suggestions_en
    
    level_key = "Mild"
    if "Moderate" in severity: level_key = "Moderate"
    elif "Severe" in severity or "High" in severity: level_key = "Severe/High"
    elif "Minimal" in severity or "Low" in severity: level_key = "Mild" 

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

# --- SIDEBAR PROFILE ---
st.sidebar.header(t['sidebar_title'])

def get_safe_index(options, default_idx=0):
    if st.session_state.reset: return 0
    return min(default_idx, len(options) - 1)

# Dynamic Inputs
age_display = ['18-22', '23-26', '27-30', 'Above 30']
age_input = st.sidebar.selectbox(t['age'], age_display, index=get_safe_index(age_display, 0))

gender_idx = st.sidebar.selectbox(t['gender'], t['genders'], index=get_safe_index(t['genders'], 0))
gender_model = 'Male' if gender_idx in ['Male', 'পুরুষ'] else 'Female'

uni_idx = st.sidebar.selectbox(t['uni'], t['unis'], index=get_safe_index(t['unis'], 1))
uni_model = 'Public' if uni_idx in ['Public', 'পাবলিক'] else 'Private'

dept_idx = st.sidebar.selectbox(t['dept'], t['depts'], index=get_safe_index(t['depts'], 0))
dept_map = {"সিএসই": "CSE", "ইইই": "EEE", "বিবিএ": "BBA", "ইংরেজি": "English", "আইন": "Law", "ফার্মাসি": "Pharmacy", "অন্যান্য": "Other"}
dept_model = dept_map.get(dept_idx, dept_idx)

year_idx = st.sidebar.selectbox(t['year'], t['years'], index=get_safe_index(t['years'], 0))
year_map = {'১ম বর্ষ': 'First Year', '২য় বর্ষ': 'Second Year', '৩য় বর্ষ': 'Third Year', '৪র্থ বর্ষ': 'Fourth Year', 'মাস্টার্স': 'Master'}
year_model = year_map.get(year_idx, year_idx)

cgpa_val = 0.00 if st.session_state.reset else 3.50
cgpa_input = st.sidebar.number_input(t['cgpa'], min_value=0.00, max_value=4.00, value=cgpa_val, step=0.01, format="%.2f")

sch_idx = st.sidebar.selectbox(t['scholarship'], t['scholars'], index=get_safe_index(t['scholars'], 1))
sch_model = 'Yes' if sch_idx in ['Yes', 'হ্যাঁ'] else 'No'

st.sidebar.markdown("---")

# Helpline
with st.sidebar.expander(t['helpline_title'], expanded=True):
    st.markdown("""
    📞 **Kaan Pete Roi:** 01779554391  
    📞 **Moner Bondhu:** 01779632588  
    🚑 **National Emergency:** 999
    """)

# --- QUESTIONNAIRE ---
st.subheader(t['section_title'])
st.info(t['instructions'])
st.caption(t['scale_caption'])

slider_options = t['slider_opts'] 
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
        key_name = f"q_{i}_{st.session_state.reset}_{lang}"
        val = st.select_slider(label=f"**{q_text}**", options=slider_options, value=slider_options[0], key=key_name)
        answers_map[i] = options_map[val]
        st.write("") 

final_answers = [answers_map[i] for i in range(26)]

if st.session_state.reset:
    st.session_state.reset = False

# --- PREDICTION ---
st.markdown("---")
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
            
            # --- GLOBAL SAFETY ALERT (Top) ---
            if final_answers[25] >= 2:
                st.markdown(f"""
                <div class="emergency-box">
                    <h3>🚨 {'Emergency Alert' if lang=='English' else 'জরুরি সতর্কতা'}</h3>
                    <p>{'Your response indicates distress. Please seek professional help or call the helpline immediately.' if lang=='English' else 'আপনার উত্তর মানসিক যন্ত্রণার ইঙ্গিত দিচ্ছে। দয়া করে অবিলম্বে পেশাদার সাহায্য নিন অথবা হেল্পলাইনে কল করুন।'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success(t['success'])
            st.subheader(t['result_title'])
            
            result_cols = st.columns(3)
            conditions = ['Anxiety', 'Stress', 'Depression']
            risk_scores = []
            healthy_count = 0
            
            report_text = f"--- MENTAL HEALTH ASSESSMENT REPORT ---\n"
            report_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_text += f"Profile: {age_input}, {gender_model}, {dept_model}\n"
            report_text += "---------------------------------------\n\n"
            
            for i, cond in enumerate(conditions):
                prob_arr = probs[i][0]
                best_idx = np.argmax(prob_arr)
                label = encoders[f'{cond} Label'].inverse_transform([best_idx])[0]
                confidence = prob_arr[best_idx] * 100
                
                # Display Logic
                display_label = label
                is_healthy = any(safe in label for safe in ["Minimal", "Low", "None", "No Depression"])
                
                # Translate Labels
                if lang == 'Bangla':
                    if is_healthy: display_label = "ঝুঁকি নেই / সুস্থ"
                    elif "Severe" in label: display_label = "তীব্র ঝুঁকি (Severe)"
                    elif "Moderate" in label: display_label = "মাঝারি ঝুঁকি (Moderate)"
                    elif "Mild" in label: display_label = "মৃদু ঝুঁকি (Mild)"
                else:
                    if label == "Minimal Anxiety": display_label = "No Anxiety / Healthy"
                    if label == "Low Stress": display_label = "No Stress / Healthy"
                    if label in ["No Depression", "Minimal Depression"]: display_label = "No Depression / Healthy"

                report_text += f"{cond}: {label} (Confidence: {confidence:.1f}%)\n"
                
                with result_cols[i]:
                    st.markdown(f"#### {cond}")
                    if is_healthy:
                        st.success(f"**{display_label}**")
                        st.progress(0)
                        healthy_count += 1
                        risk_scores.append((cond, 0, label))
                        
                        # --- MODIFICATION: Clinical Note inside Depression Card ---
                        # Logic: If Healthy/Low Risk BUT Self-harm (Index 25) is High
                        if cond == 'Depression' and final_answers[25] >= 2:
                            warn_text = "⚠️ **Clinical Note:** Self-harm risk detected despite low overall score." if lang == 'English' else "⚠️ **ক্লিনিক্যাল নোট:** সামগ্রিক স্কোর কম হলেও আত্মহানির ঝুঁকি লক্ষ্য করা যাচ্ছে।"
                            st.warning(warn_text)

                    else:
                        st.error(f"**{display_label}**")
                        st.progress(int(confidence))
                        st.caption(f"{'Risk Probability' if lang=='English' else 'ঝুঁকির সম্ভাবনা'}: {confidence:.1f}%")
                        risk_scores.append((cond, confidence, label))

            st.markdown("---")
            
            # --- PROFESSIONAL RECOMMENDATIONS ---
            st.subheader(t['suggestions'])
            
            if healthy_count == 3:
                st.balloons()
                st.markdown(t['healthy_msg'])
                report_text += "\nRecommendation: Maintain current healthy lifestyle."
            else:
                risk_scores.sort(key=lambda x: x[1], reverse=True)
                
                for cond, conf, severity_label in risk_scores:
                    if conf > 0: 
                        box_class = "suggestion-warning" if "Severe" in severity_label else "suggestion-box"
                        st.markdown(f"##### 👉 **{cond} ({severity_label})**")
                        suggestions = get_professional_suggestions(cond, severity_label, lang)
                        for tip in suggestions:
                            st.info(tip)
                        report_text += f"\n[{cond} Suggestions]:\n" + "\n".join([s.replace('**', '') for s in suggestions]) + "\n"

            # Download Report
            st.markdown("---")
            report_text += "\n---------------------------------------\n"
            report_text += "DISCLAIMER: This result is based on ML patterns and is not a clinical diagnosis."
            
            st.download_button(
                label=t['download_btn'],
                data=report_text,
                file_name=f"Assessment_Report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Feature column count mismatch!")

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.divider()

st.markdown(f"""
<div class='footer'>
    <div style="font-weight: bold; margin-bottom: 5px;">{t['dev_by']}</div>
    <div style="font-size: 16px; color: #333; font-weight: 600;">Team Dual Core</div>
    <div style="font-size: 14px; margin-top: 5px;">Ovi Sarker & BM Sabbir Hossen Riad</div>
    <div style="font-size: 12px; color: #777; margin-top: 5px;">Department of CSE, Daffodil International University</div>
    <br>
    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; display: inline-block; border: 1px solid #ffeeba;">
        <span style="font-size: 11px; color: #856404;">
            {t['disclaimer_short']}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
